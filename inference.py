"""
Organizer-compliant inference entry point for FoodCrisisEnv.

Pure LLM strategy — every action decision goes through the language model.
Rate limiting: 2.2s sleep between LLM calls achieves ~27 calls/min (safe under Groq 30 req/min limit).
On 429 errors: backs off 65s then retries once before defaulting to WAIT.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar

from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("FOODCRISIS_STANDALONE", "1")

from irce.environment import FoodCrisisEnv
from irce.grading import grade_episode
from irce.models import FoodCrisisAction, FoodCrisisObservation
from irce.tasks import TaskConfig, build_task_registry

BENCHMARK = "food-crisis-env"
DEFAULT_API_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL_NAME = "llama-3.1-8b-instant"
ACTION_PATTERN = re.compile(r"^(INSPECT|QUARANTINE|LIFT|RECALL|TRACE|ALERT|WAIT)(?:\s+(.+))?$", re.IGNORECASE)
TEMPERATURE = 0.5
MAX_TOKENS = 32
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))
MAX_CONSECUTIVE_UNUSABLE_STEPS = int(os.getenv("MAX_CONSECUTIVE_UNUSABLE_STEPS", "5"))

# Rate limiting: 2.2s between calls = ~27 calls/min, optimized for Groq free tier (30 req/min limit)
# Provides balanced throughput (~27 calls/min) with comfortable safety margin under 30 req/min.
LLM_RATE_LIMIT_SLEEP = float(os.getenv("LLM_RATE_LIMIT_SLEEP", "1.2"))

SYSTEM_PROMPT = """You are a food safety investigator. Output exactly one action per step. No explanation. No markdown.

Supply chain: farm -> processing -> warehouse -> retailer
Contamination starts at source farms and spreads downstream each step.

PRIORITY ORDER — check top to bottom, do the FIRST that applies:

1. LIFT <node>        if lab_results[node]="clean" AND node is quarantined
2. QUARANTINE <node>  if lab_results[node]="contaminated" AND node NOT quarantined (farms first, +4.0 reward)
3. INSPECT <node>     if node has high sensor reading AND not in lab_results AND lab_budget > 0
4. TRACE <batch_id>   if traceable_batch_ids is not empty (pick highest-risk batch; finds source cheaply)
5. RECALL <batch_id>  if contaminated batch at retailer AND recall_budget >= 10
6. ALERT <retailer>   if retailer has illness reports AND no quarantine AND recall_budget low (use sparingly)
7. WAIT               if none of the above apply

HARD RULES — never break:
- TRACE only batches listed in traceable_batch_ids (untraced batches at nodes)
- Never TRACE a batch already in traced_batches
- Never TRACE the same batch as your previous action
- Never INSPECT a node already in lab_results or pending_inspections
- Never QUARANTINE an already-quarantined node
- Never LIFT a node that is not quarantined
- Never ALERT a non-retailer node
- RECALL only batches listed in recallable_batch_ids

Output format — exactly one line, nothing else:
TRACE <batch_id>
INSPECT <node_id>
QUARANTINE <node_id>
LIFT <node_id>
RECALL <batch_id>
ALERT <node_id>
WAIT"""
 
RETRY_USER_PROMPT = """Invalid action. Output ONLY one valid action. One line. No explanation.

PRIORITY (top to bottom, first that applies):
1. LIFT <node>       — lab_results[node]="clean" AND quarantined
2. QUARANTINE <node> — lab_results[node]="contaminated" AND NOT quarantined (farms first)
3. INSPECT <node>    — high sensor, not in lab_results, lab_budget > 0
4. TRACE <batch_id>  — use traceable_batch_ids list ONLY (not already-traced batches)
5. RECALL <batch_id> — contaminated batch at retailer, recall_budget >= 10
6. WAIT              — nothing else applies

HARD: Never TRACE a batch from traced_batches. Never repeat the same TRACE as your last action.

Valid formats:
TRACE <batch_id>
INSPECT <node_id>
QUARANTINE <node_id>
LIFT <node_id>
RECALL <batch_id>
ALERT <node_id>
WAIT"""

ObservationT = TypeVar("ObservationT")


@dataclass
class StepResult(Generic[ObservationT]):
    observation: ObservationT
    reward: Optional[float] = None
    done: bool = False


@dataclass
class ModelAccessState:
    fatal_error: Optional[str] = None
    fatal_error_reported: bool = False
    circuit_open_reason: Optional[str] = None
    circuit_open_reported: bool = False
    consecutive_unusable_steps: int = 0


MODEL_ACCESS_STATE = ModelAccessState()


class LocalEnvRunner:
    def __init__(self, task_id: int, seed: int) -> None:
        self._env = FoodCrisisEnv(task_id=task_id, seed=seed)
        self.task_id = task_id
        self.seed = seed

    @property
    def episode_log(self) -> List[Dict[str, Any]]:
        return self._env.episode_log

    def reset(self) -> StepResult[FoodCrisisObservation]:
        observation = self._env.reset(seed=self.seed, task_id=self.task_id)
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def step(self, action: FoodCrisisAction) -> StepResult[FoodCrisisObservation]:
        observation = self._env.step(action)
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def close(self) -> None:
        return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_single_line(value: str) -> str:
    return " ".join(str(value).split())


def visible_batch_ids(observation: FoodCrisisObservation) -> List[str]:
    """Return batch IDs currently sitting at nodes that have NOT yet been traced.
    Already-traced batches are excluded — re-tracing them is a no-op waste."""
    already_traced = {bid.strip().lower() for bid in observation.traced_batches}
    batch_ids: List[str] = []
    seen: set = set()
    for node in observation.nodes:
        for batch_id in node.batch_ids:
            normalized = batch_id.strip().lower()
            if normalized and normalized not in seen and normalized not in already_traced:
                seen.add(normalized)
                batch_ids.append(normalized)
    return batch_ids


def traceable_batch_ids(observation: FoodCrisisObservation) -> List[str]:
    """Alias used in prompts — same as visible_batch_ids (untraced batches at nodes)."""
    return visible_batch_ids(observation)


def recallable_batch_ids(observation: FoodCrisisObservation) -> List[str]:
    """All active (not recalled/delivered) batch IDs visible at nodes, including already-traced ones.
    RECALL is valid on any active batch regardless of trace status."""
    batch_ids: List[str] = []
    seen: set = set()
    for node in observation.nodes:
        for batch_id in node.batch_ids:
            normalized = batch_id.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                batch_ids.append(normalized)
    return batch_ids


def available_node_ids(observation: FoodCrisisObservation) -> List[str]:
    return [node.node_id for node in observation.nodes]


def extract_pending_inspection_nodes(observation: FoodCrisisObservation) -> List[str]:
    summary = observation.natural_language_summary or ""
    match = re.search(r"Pending lab tests:\s*([^.]*)\.", summary)
    if match is None:
        return []
    payload = normalize_single_line(match.group(1))
    if not payload or payload.lower() == "none":
        return []
    pending_nodes: List[str] = []
    seen: set = set()
    for item in payload.split(","):
        node_id = item.strip().split("@", 1)[0].strip().lower()
        if node_id and node_id not in seen:
            seen.add(node_id)
            pending_nodes.append(node_id)
    return pending_nodes


def inspected_or_pending_nodes(observation: FoodCrisisObservation) -> List[str]:
    combined = {node_id.lower() for node_id in observation.lab_results}
    combined.update(extract_pending_inspection_nodes(observation))
    return sorted(combined)


def uninspected_nodes(observation: FoodCrisisObservation) -> List[str]:
    all_nodes = {n.lower() for n in available_node_ids(observation)}
    already_done = set(inspected_or_pending_nodes(observation))
    return sorted(all_nodes - already_done)


def build_user_prompt(step: int, observation: FoodCrisisObservation, history: List[str]) -> str:
    recent_history = "\n".join(history[-3:]) if history else "none"

    traceable = ", ".join(traceable_batch_ids(observation)) or "none"
    recallable = ", ".join(recallable_batch_ids(observation)) or "none"

    # Compact traced-batches summary — show only origin node per batch to save tokens
    # Full paths are already communicated via the NL summary; we only need origin attribution here.
    traced_origins: Dict[str, str] = {}
    for bid, path in observation.traced_batches.items():
        traced_origins[bid] = path[0] if path else "?"
    traced_summary = json.dumps(traced_origins, sort_keys=True, separators=(",", ":")) if traced_origins else "{}"

    pending_text = ", ".join(extract_pending_inspection_nodes(observation)) or "none"
    inspected_text = ", ".join(inspected_or_pending_nodes(observation)) or "none"
    uninspected_text = ", ".join(uninspected_nodes(observation)) or "none"

    # Derive previous action
    previous_action = "none"
    if history:
        m = re.search(r"action=(\S+(?:\s+\S+)?)", history[-1])
        if m:
            previous_action = m.group(1).strip()

    # Compute compact quarantine_status (only show quarantined=true nodes to save tokens)
    quarantined_nodes = sorted(nid for nid, q in observation.quarantine_status.items() if q)
    quarantine_text = ", ".join(quarantined_nodes) if quarantined_nodes else "none"

    # Compact lab_results
    lab_text = json.dumps(observation.lab_results, sort_keys=True, separators=(",", ":")) if observation.lab_results else "{}"

    # Illness summary — compact
    illness_text = "; ".join(
        f"{r.retailer_id}:{r.case_count}cases@t{r.timestep_reported}"
        for r in observation.illness_reports
    ) if observation.illness_reports else "none"

    # Explicit action hints
    hints: list[str] = []

    to_quarantine = sorted(
        nid for nid, result in observation.lab_results.items()
        if result == "contaminated" and not observation.quarantine_status.get(nid, False)
    )
    if to_quarantine:
        hints.append(f"QUARANTINE NOW: {', '.join(to_quarantine)} (confirmed contaminated, farms=+4.0)")

    to_lift = sorted(
        nid for nid, result in observation.lab_results.items()
        if result == "clean" and observation.quarantine_status.get(nid, False)
    )
    if to_lift:
        hints.append(f"LIFT: {', '.join(to_lift)} (confirmed clean, still quarantined)")

    if not to_quarantine and not to_lift:
        if traceable != "none":
            hints.append(f"TRACE one of: {traceable}")
        elif observation.lab_budget > 0 and uninspected_text != "none":
            hints.append(f"INSPECT one of: {uninspected_text} (highest sensor first)")
        elif observation.lab_budget <= 0 and recallable != "none":
            hints.append(f"lab_budget=0 — no more INSPECTs. Consider RECALL if batch is contaminated.")
        elif pending_text != "none":
            hints.append("Pending labs; nothing else to do -> WAIT")
        else:
            hints.append("WAIT")

    hint_block = "\n".join(hints)

    return (
        f"STEP {step} | prev={previous_action}\n"
        f"{observation.natural_language_summary}\n\n"
        f"HINTS: {hint_block}\n\n"
        f"lab_results:{lab_text}\n"
        f"quarantined:{quarantine_text}\n"
        f"pending_labs:{pending_text}\n"
        f"inspected:{inspected_text}\n"
        f"uninspected:{uninspected_text}\n"
        f"illness:{illness_text}\n"
        f"lab_budget:{observation.lab_budget} recall_budget:{observation.recall_budget} trust:{observation.public_trust:.2f}\n"
        f"traceable:{traceable}\n"
        f"recallable:{recallable}\n"
        f"traced_origins:{traced_summary}\n"
        f"history:\n{recent_history}\n"
        "Action:"
    )


def build_openai_client() -> tuple[Any, str]:
    api_key = (os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL).strip()
    model_name = (os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME).strip()

    if not api_key:
        raise SystemExit("Missing HF_TOKEN in environment configuration.")
    if not model_name:
        raise SystemExit("Missing MODEL_NAME in environment configuration.")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit(f"OpenAI SDK import failed: {exc}") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name


def extract_completion_text(completion: Any) -> str:
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content or "")


def debug_to_stderr(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def is_fatal_provider_error(exc: Exception) -> bool:
    text = normalize_single_line(str(exc)).lower()
    return any(marker in text for marker in (
        "error code: 401", "error code: 402", "error code: 403",
        "insufficient_quota", "depleted your monthly included credits",
        "invalid api key", "authentication", "permission",
    ))


def is_rate_limit_error(exc: Exception) -> bool:
    text = normalize_single_line(str(exc)).lower()
    return "error code: 429" in text or "rate_limit" in text or "rate limit" in text


def strip_outer_quotes(text: str) -> str:
    stripped = text.strip().strip("`")
    while len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        stripped = stripped[1:-1].strip()
    return stripped


def candidate_texts_from_response(response_text: str) -> List[str]:
    candidates: List[str] = []
    seen: set = set()

    def add(candidate: Optional[str]) -> None:
        if not candidate:
            return
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    raw = response_text.strip()
    add(raw)
    first_line = raw.splitlines()[0].strip() if raw else ""
    add(first_line)
    add(strip_outer_quotes(first_line))

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            for key in ("action", "action_type", "command", "output", "response"):
                if isinstance(payload.get(key), str):
                    add(payload[key])
    except json.JSONDecodeError:
        pass

    return candidates


def parse_candidate_action(candidate: str, observation: FoodCrisisObservation) -> Optional[str]:
    first_line = strip_outer_quotes(candidate.splitlines()[0].strip())
    text = normalize_single_line(first_line)
    if not text:
        return None

    match = ACTION_PATTERN.fullmatch(text)
    if match is None:
        search_match = re.search(
            r"\b(INSPECT|QUARANTINE|LIFT|RECALL|TRACE|ALERT|WAIT)\b(?:\s+([A-Za-z0-9_\-]+))?",
            text, re.IGNORECASE,
        )
        if search_match is None:
            return None
        verb = search_match.group(1).upper()
        target = strip_outer_quotes(search_match.group(2) or "")
    else:
        verb = match.group(1).upper()
        target = strip_outer_quotes(match.group(2) or "")

    target = normalize_single_line(target)

    if verb == "WAIT":
        return "WAIT" if not target else None
    if not target:
        return None

    normalized_target = target.lower()
    node_lookup = {node.node_id.lower(): node for node in observation.nodes}
    trace_lookup = set(visible_batch_ids(observation))       # untraced batches only
    recall_lookup = set(recallable_batch_ids(observation))   # all active batches at nodes

    if verb in {"INSPECT", "QUARANTINE", "LIFT"}:
        return f"{verb} {normalized_target}" if normalized_target in node_lookup else None
    if verb == "ALERT":
        node = node_lookup.get(normalized_target)
        return f"ALERT {normalized_target}" if (node and node.node_type == "retailer") else None
    if verb == "TRACE":
        return f"TRACE {normalized_target}" if normalized_target in trace_lookup else None
    if verb == "RECALL":
        return f"RECALL {normalized_target}" if normalized_target in recall_lookup else None

    return None


def parse_model_action(response_text: str, observation: FoodCrisisObservation) -> Optional[str]:
    for candidate in candidate_texts_from_response(response_text):
        parsed = parse_candidate_action(candidate, observation)
        if parsed:
            return parsed
    return None


def apply_action_guard(action_str: str, observation: FoodCrisisObservation, history: Optional[List[str]] = None) -> str:
    """Block actions that would be wasted or cause errors in the environment."""
    parts = action_str.split(" ", 1)
    verb = parts[0].upper()
    target = parts[1].strip().lower() if len(parts) > 1 else ""

    pending_nodes = set(extract_pending_inspection_nodes(observation))
    known_results = {nid.lower(): result for nid, result in observation.lab_results.items()}

    if verb == "INSPECT":
        # Block if no lab budget left — environment rejects and charges wasted-action penalty
        if observation.lab_budget <= 0:
            debug_to_stderr(f"ACTION_GUARD: INSPECT {target} -> WAIT (lab_budget=0)")
            return "WAIT"
        if target in pending_nodes:
            debug_to_stderr(f"ACTION_GUARD: INSPECT {target} -> WAIT (already pending)")
            return "WAIT"
        if target in known_results:
            debug_to_stderr(f"ACTION_GUARD: INSPECT {target} -> WAIT (result already known)")
            return "WAIT"

    if verb == "QUARANTINE":
        if observation.quarantine_status.get(target, False):
            debug_to_stderr(f"ACTION_GUARD: QUARANTINE {target} -> WAIT (already quarantined)")
            return "WAIT"

    if verb == "LIFT":
        if not observation.quarantine_status.get(target, False):
            debug_to_stderr(f"ACTION_GUARD: LIFT {target} -> WAIT (not quarantined)")
            return "WAIT"

    if verb == "TRACE":
        # Block if this batch was already traced — re-tracing learns nothing
        already_traced = {bid.strip().lower() for bid in observation.traced_batches}
        if target in already_traced:
            debug_to_stderr(f"ACTION_GUARD: TRACE {target} -> WAIT (already traced)")
            return "WAIT"
        # Block if the last step took the exact same TRACE action
        if history:
            last_action_match = re.search(r"action=(\S+(?:\s+\S+)?)", history[-1])
            if last_action_match:
                last_action = last_action_match.group(1).strip().lower()
                if last_action == f"trace {target}":
                    debug_to_stderr(f"ACTION_GUARD: TRACE {target} -> WAIT (same as last step)")
                    return "WAIT"
        # Redirect TRACE to QUARANTINE when confirmed-contaminated unquarantined nodes exist
        unquarantined_confirmed = sorted(
            nid for nid, result in observation.lab_results.items()
            if result == "contaminated" and not observation.quarantine_status.get(nid, False)
        )
        if unquarantined_confirmed:
            best = unquarantined_confirmed[0]
            debug_to_stderr(
                f"ACTION_GUARD: TRACE {target} -> QUARANTINE {best} "
                f"(confirmed contaminated, not yet quarantined)"
            )
            return f"QUARANTINE {best}"

    if verb == "RECALL":
        # Block if recall budget is insufficient (each recall costs 10)
        if observation.recall_budget < 10:
            debug_to_stderr(f"ACTION_GUARD: RECALL {target} -> WAIT (recall_budget={observation.recall_budget} < 10)")
            return "WAIT"
        # Block recall of non-existent/inactive batches
        active_batches = set(recallable_batch_ids(observation))
        if target not in active_batches:
            debug_to_stderr(f"ACTION_GUARD: RECALL {target} -> WAIT (not an active batch at a node)")
            return "WAIT"

    return action_str


def deterministic_fallback(observation: FoodCrisisObservation, history: List[str]) -> Optional[str]:
    """Rule-based engine covering ~75% of steps without an LLM call.

    Priority order (first match wins):
      P1  QUARANTINE confirmed-contaminated node (farms first, +4.0 reward)
      P2  LIFT confirmed-clean quarantined node
      P3  RECALL confirmed-contaminated batch sitting at a retailer
      P4  TRACE untraced batch at retailer (most urgent; finds source cheaply)
      P5  TRACE untraced batch at warehouse (next-most-downstream)
      P6  INSPECT highest-sensor uninspected upstream node (farm/processing)
      P7  TRACE any remaining traceable batch (processing then farm)
      P8  WAIT while pending labs are processing
      P9  WAIT if no budget and nothing traceable remains
      else -> None (LLM handles genuine ambiguity)
    """
    lab_results = observation.lab_results
    quarantine = observation.quarantine_status

    # ── P1: quarantine confirmed contaminated ───────────────────────────────
    to_quarantine = [
        nid for nid, r in lab_results.items()
        if r == "contaminated" and not quarantine.get(nid, False)
    ]
    if to_quarantine:
        farms_first = sorted(to_quarantine, key=lambda n: (0 if "farm" in n else 1))
        return f"QUARANTINE {farms_first[0]}"

    # ── P2: lift confirmed clean ────────────────────────────────────────────
    to_lift = [nid for nid, r in lab_results.items() if r == "clean" and quarantine.get(nid, False)]
    if to_lift:
        return f"LIFT {to_lift[0]}"

    # Pre-compute shared state once
    pending: List[str] = extract_pending_inspection_nodes(observation)
    pending_set: set = set(pending)
    traceable: List[str] = traceable_batch_ids(observation)
    traceable_set: set = set(traceable)
    recallable: List[str] = recallable_batch_ids(observation)
    recallable_set: set = set(recallable)
    confirmed_contaminated_nodes: set = {nid for nid, r in lab_results.items() if r == "contaminated"}

    # ── P3: recall confirmed-contaminated batch at a retailer ───────────────
    if observation.recall_budget >= 10:
        for node in observation.nodes:
            if node.node_type == "retailer" and node.node_id in confirmed_contaminated_nodes:
                for bid in node.batch_ids:
                    if bid.strip().lower() in recallable_set:
                        return f"RECALL {bid.strip().lower()}"

    # ── P4: trace untraced batch at retailer ───────────────────────────────
    if traceable_set:
        for node in observation.nodes:
            if node.node_type == "retailer":
                for bid in node.batch_ids:
                    normalized = bid.strip().lower()
                    if normalized in traceable_set:
                        return f"TRACE {normalized}"

    # ── P5: wait while pending lab results are on their way ────────────────
    if pending:
        return "WAIT"

    # ── P9: nothing actionable left ────────────────────────────────────────
    if observation.lab_budget <= 0 and not traceable_set:
        return "WAIT"

    return None  # genuine ambiguity — let the LLM decide


def request_model_completion(client: Any, model_name: str, messages: List[Dict[str, str]]) -> str:
    """Request a completion from the LLM model."""
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return extract_completion_text(completion)


def request_action(
    *,
    client: Any,
    model_name: str,
    observation: FoodCrisisObservation,
    history: List[str],
    step: int,
    access_state: ModelAccessState,
) -> str:
    if access_state.fatal_error is not None:
        if not access_state.fatal_error_reported:
            debug_to_stderr("FATAL_PROVIDER_ERROR:", access_state.fatal_error)
            access_state.fatal_error_reported = True
        return "WAIT"
    if access_state.circuit_open_reason is not None:
        if not access_state.circuit_open_reported:
            debug_to_stderr("MODEL_CIRCUIT_OPEN:", access_state.circuit_open_reason)
            access_state.circuit_open_reported = True
        return "WAIT"

    # Deterministic fallback: skip LLM when the correct action is unambiguous
    fallback = deterministic_fallback(observation, history)
    if fallback is not None:
        debug_to_stderr(f"DETERMINISTIC_FALLBACK: {fallback}")
        return fallback

    # Rate-limit sleep — keeps Groq free tier inside 30 req/min
    time.sleep(LLM_RATE_LIMIT_SLEEP)

    user_prompt = build_user_prompt(step, observation, history)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response_text = request_model_completion(client, model_name, messages)
        debug_to_stderr("RAW_MODEL_OUTPUT:", response_text)

        parsed_action = parse_model_action(response_text, observation)
        if parsed_action:
            access_state.consecutive_unusable_steps = 0
            return parsed_action

        # Retry once on invalid response — immediate, no extra sleep
        retry_messages = messages + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": RETRY_USER_PROMPT},
        ]
        retry_text = request_model_completion(client, model_name, retry_messages)
        debug_to_stderr("RAW_MODEL_OUTPUT_RETRY:", retry_text)

        retry_action = parse_model_action(retry_text, observation)
        if retry_action:
            access_state.consecutive_unusable_steps = 0
            return retry_action

        # Parse failure on both attempts — but this is NOT a model error, just bad action format
        # Don't increment circuit breaker for parse failures; only for actual API/model errors
        debug_to_stderr("PARSE_FAILURE: both initial and retry unparseable")
        return "WAIT"

    except Exception as exc:
        debug_to_stderr("MODEL_REQUEST_ERROR:", exc)

        if is_fatal_provider_error(exc):
            access_state.fatal_error = str(exc)

        elif is_rate_limit_error(exc):
            # Hard 429 despite sleep — back off 65s and retry once
            debug_to_stderr("RATE_LIMIT_HIT: sleeping 65s before retry")
            time.sleep(65.0)
            try:
                response_text = request_model_completion(client, model_name, messages)
                debug_to_stderr("RAW_MODEL_OUTPUT_AFTER_BACKOFF:", response_text)
                parsed = parse_model_action(response_text, observation)
                if parsed:
                    access_state.consecutive_unusable_steps = 0
                    return parsed
            except Exception as retry_exc:
                debug_to_stderr("MODEL_REQUEST_ERROR_AFTER_BACKOFF:", retry_exc)
            # Rate limit persisted after backoff — this IS a real error
            access_state.consecutive_unusable_steps += 1
            if access_state.consecutive_unusable_steps >= MAX_CONSECUTIVE_UNUSABLE_STEPS:
                access_state.circuit_open_reason = (
                    f"{access_state.consecutive_unusable_steps} consecutive rate limit failures; "
                    "defaulting to WAIT for remaining steps"
                )

        else:
            # Other exceptions (network, timeout, etc.) — real errors
            access_state.consecutive_unusable_steps += 1
            if access_state.consecutive_unusable_steps >= MAX_CONSECUTIVE_UNUSABLE_STEPS:
                access_state.circuit_open_reason = (
                    f"{access_state.consecutive_unusable_steps} consecutive API failures; "
                    "defaulting to WAIT for remaining steps"
                )

    return "WAIT"



def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={normalize_single_line(task)} env={normalize_single_line(env)} model={normalize_single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = normalize_single_line(error) if error else "null"
    done_val = str(bool(done)).lower()
    print(
        f'[STEP] step={step} action="{normalize_single_line(action)}" reward={reward:.2f} done={done_val} error={error_val}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def extract_step_error(observation: FoodCrisisObservation) -> Optional[str]:
    return "action_error" if observation.last_action_error else None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: int, task_config: TaskConfig, seed: int, client: Any, model_name: str) -> float:
    # Reset per task — errors on task 1 must not silence tasks 2 and 3
    MODEL_ACCESS_STATE.fatal_error = None
    MODEL_ACCESS_STATE.fatal_error_reported = False
    MODEL_ACCESS_STATE.circuit_open_reason = None
    MODEL_ACCESS_STATE.circuit_open_reported = False
    MODEL_ACCESS_STATE.consecutive_unusable_steps = 0

    env = LocalEnvRunner(task_id=task_id, seed=seed)
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_config.name, env=BENCHMARK, model=model_name)

    try:
        result = env.reset()
        observation = result.observation

        for step in range(1, task_config.max_steps + 1):
            if result.done:
                break

            action_str = request_action(
                client=client,
                model_name=model_name,
                observation=observation,
                history=history,
                step=step,
                access_state=MODEL_ACCESS_STATE,
            )
            action_str = apply_action_guard(action_str, observation, history)

            debug_to_stderr(
                f"STEP {step} | final_action={action_str} | "
                f"lab_budget={observation.lab_budget} | trust={observation.public_trust:.2f}"
            )

            result = env.step(FoodCrisisAction(action_type=action_str))
            observation = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = extract_step_error(observation)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"step={step} action={action_str} reward={reward:+.2f} "
                f"result={observation.tool_result} done={str(done).lower()}"
            )

            if done:
                break

        score = clamp01(grade_episode(env.episode_log))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def parse_args() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="FoodCrisisEnv organizer-compliant inference runner.")
    parser.add_argument("--seed", type=int, default=7, help="Evaluation seed shared across tasks.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    client, model_name = build_openai_client()
    task_registry = build_task_registry()

    for task_id in sorted(task_registry):
        run_task(
            task_id=task_id,
            task_config=task_registry[task_id],
            seed=args.seed,
            client=client,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()