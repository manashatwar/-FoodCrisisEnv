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
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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
DEFAULT_MODEL_NAME ="llama-3.1-8b-instant"
ACTION_PATTERN = re.compile(r"^(INSPECT|QUARANTINE|LIFT|RECALL|TRACE|ALERT|WAIT)(?:\s+(.+))?$", re.IGNORECASE)
TEMPERATURE = 0.0
MAX_TOKENS = 32
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))
MAX_CONSECUTIVE_UNUSABLE_STEPS = int(os.getenv("MAX_CONSECUTIVE_UNUSABLE_STEPS", "5"))

# Rate limiting: 2.2s between calls = ~27 calls/min, optimized for Groq free tier (30 req/min limit)
# Provides balanced throughput (~27 calls/min) with comfortable safety margin under 30 req/min.
LLM_RATE_LIMIT_SLEEP = float(os.getenv("LLM_RATE_LIMIT_SLEEP", "2.2"))

SYSTEM_PROMPT = """You are a food safety incident responder controlling FoodCrisisEnv.
Your job: contain contamination quickly while preserving public trust and limited budgets.

CRITICAL: This is a tracing task. Find the SOURCE farm, not just reactive symptoms.
- Quarantining source farm: +4.0 reward (high value!)
- Quarantining downstream contaminated node: +2.0 reward (half the value)
- Wrong quarantine: -2.0 + trust damage (very costly)
- TRACE is cheap (-0.1 cost) and reveals source — use it early and often

Supply chain flow: farm -> processing -> warehouse -> retailer
Contamination starts at source(s) and spreads downstream each step. Once at retail, it's exposure and illness cases.

===== DECISION FLOW (Priority Order) =====

At each step, follow this priority exactly:

1. LIFT first: Any node in lab_results="clean" AND currently quarantined? -> LIFT it
   (Restores supply chain, builds trust, no cost)

2. QUARANTINE second: Any node in lab_results="contaminated" AND NOT quarantined? -> QUARANTINE it
   (Prioritize farms (source) over processing over warehouse — upstream first)
   (Farms: +4.0 reward, others: +2.0)

3. TRACE third: visible_batch_ids NOT empty AND (illness_reports present OR high sensor nodes exist)?
   -> TRACE the most exposed batch (from retailer if possible, else highest risk)
   (Cheap (-0.1) way to find source origin and path; low risk)

4. INSPECT fourth: Uninspected high-risk nodes AND lab_budget > 0?
   -> INSPECT the most suspicious node (high sensor, near illness, or upstream of contamination)
   (Costs lab_budget=1; use selectively on ambiguous nodes)

5. RECALL fifth: Contaminated batch already at retailer/warehouse AND recall_budget >= 10?
   -> RECALL it (prevents further exposure; +1.5 reward for correct recall)
   (Only when you're confident batch is contaminated)

6. ALERT sixth: Retailer with active illness AND no quarantine/alert yet AND still waiting?
   -> ALERT it (buys 3 steps, costs -0.08 trust, but slows exposure)
   (Use sparingly; only when necessary to delay while waiting for lab results)

7. WAIT: Only when nothing above applies
   (Safest action when uncertain; minimal cost)

===== CRITICAL RULES — NEVER BREAK THESE =====
* visible_batch_ids is ground truth. ONLY TRACE/RECALL batches you see here.
* visible_batch_ids = batches_currently_at_nodes + already_traced_batches
* Do NOT hallucinate batch IDs. If it's not in visible_batch_ids, don't TRACE it.
* Do NOT repeat the same action two steps in a row (unless new lab results just landed).
* Do NOT INSPECT a node already in lab_results or pending inspection.
* Do NOT QUARANTINE an already-quarantined node.
* Do NOT LIFT a node that is not quarantined.
* Do NOT ALERT a non-retailer node.

===== TRACING STRATEGY (Why it matters) =====
Early TRACE wins the game:
- TRACE reveals the origin farm and full path of a batch
- Once you know the source, INSPECT it to confirm, then QUARANTINE it (+4.0)
- This stops contamination at the root, not downstream
- Cost: only -0.1 per TRACE; ROI is huge vs blind INSPECTs

Observable signals to act on:
- Illness reports at retailers → trace those batches backward to origin
- High sensor spikes at farms → inspect to confirm, then quarantine if contaminated
- Multiple retailers sick → likely shared upstream source → trace to find it

===== OUTPUT FORMAT =====
Exactly one action. One line. No explanation. No JSON. No markdown.

TRACE <batch_id>
INSPECT <node_id>
QUARANTINE <node_id>
LIFT <node_id>
RECALL <batch_id>
ALERT <node_id>
WAIT"""
 
RETRY_USER_PROMPT = """Your previous response was not a valid action.
Output ONLY one valid action on a single line. No explanation. No JSON. No markdown.

Valid formats:
TRACE <batch_id>
INSPECT <node_id>
QUARANTINE <node_id>
LIFT <node_id>
RECALL <batch_id>
ALERT <node_id>
WAIT

PRIORITY CHECK (in order):
1. Any clean node still quarantined? -> LIFT it
2. Any contaminated node not yet quarantined? -> QUARANTINE it (farms first)
3. Any visible batch + illness exists? -> TRACE it (cheap, finds source)
4. Any high-risk uninspected node + lab_budget > 0? -> INSPECT it
5. Any contaminated batch at retailer + recall_budget >= 10? -> RECALL it
6. Otherwise -> WAIT

CRITICAL: Only use batch IDs from visible_batch_ids for TRACE/RECALL.
Never repeat the exact same action two steps in a row unless new lab results landed."""

ObservationT = TypeVar("ObservationT")


@dataclass
class StepResult(Generic[ObservationT]):
    observation: ObservationT
    reward: float | None = None
    done: bool = False


@dataclass
class ModelAccessState:
    fatal_error: str | None = None
    fatal_error_reported: bool = False
    circuit_open_reason: str | None = None
    circuit_open_reported: bool = False
    consecutive_unusable_steps: int = 0


MODEL_ACCESS_STATE = ModelAccessState()


class LocalEnvRunner:
    def __init__(self, task_id: int, seed: int) -> None:
        self._env = FoodCrisisEnv(task_id=task_id, seed=seed)
        self.task_id = task_id
        self.seed = seed

    @property
    def episode_log(self) -> list[dict[str, Any]]:
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


def visible_batch_ids(observation: FoodCrisisObservation) -> list[str]:
    batch_ids: list[str] = []
    seen: set[str] = set()
    # Batches currently at nodes
    for node in observation.nodes:
        for batch_id in node.batch_ids:
            normalized = batch_id.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                batch_ids.append(normalized)
    # Batches already traced (model can reference these)
    for batch_id in observation.traced_batches.keys():
        normalized = batch_id.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            batch_ids.append(normalized)
    return batch_ids


def available_node_ids(observation: FoodCrisisObservation) -> list[str]:
    return [node.node_id for node in observation.nodes]


def extract_pending_inspection_nodes(observation: FoodCrisisObservation) -> list[str]:
    summary = observation.natural_language_summary or ""
    match = re.search(r"Pending lab tests:\s*([^.]*)\.", summary)
    if match is None:
        return []
    payload = normalize_single_line(match.group(1))
    if not payload or payload.lower() == "none":
        return []
    pending_nodes: list[str] = []
    seen: set[str] = set()
    for item in payload.split(","):
        node_id = item.strip().split("@", 1)[0].strip().lower()
        if node_id and node_id not in seen:
            seen.add(node_id)
            pending_nodes.append(node_id)
    return pending_nodes


def inspected_or_pending_nodes(observation: FoodCrisisObservation) -> list[str]:
    combined = {node_id.lower() for node_id in observation.lab_results}
    combined.update(extract_pending_inspection_nodes(observation))
    return sorted(combined)


def uninspected_nodes(observation: FoodCrisisObservation) -> list[str]:
    all_nodes = {n.lower() for n in available_node_ids(observation)}
    already_done = set(inspected_or_pending_nodes(observation))
    return sorted(all_nodes - already_done)


def build_user_prompt(step: int, observation: FoodCrisisObservation, history: list[str]) -> str:
    recent_history = "\n".join(history[-4:]) if history else "none"
    illness_payload = [report.model_dump() for report in observation.illness_reports]
    node_ids = ", ".join(available_node_ids(observation)) or "none"
    batch_ids = ", ".join(visible_batch_ids(observation)) or "none"
    pending_text = ", ".join(extract_pending_inspection_nodes(observation)) or "none"
    inspected_pending_text = ", ".join(inspected_or_pending_nodes(observation)) or "none"
    uninspected_text = ", ".join(uninspected_nodes(observation)) or "none"

    # Explicit action hints — removes ambiguity for the model
    hints: list[str] = []

    to_quarantine = [
        nid for nid, result in observation.lab_results.items()
        if result == "contaminated" and not observation.quarantine_status.get(nid, False)
    ]
    if to_quarantine:
        hints.append(
            f"ACTION REQUIRED: Confirmed contaminated and NOT quarantined: "
            f"{', '.join(sorted(to_quarantine))} -> QUARANTINE one NOW."
        )

    to_lift = [
        nid for nid, result in observation.lab_results.items()
        if result == "clean" and observation.quarantine_status.get(nid, False)
    ]
    if to_lift:
        hints.append(
            f"ACTION AVAILABLE: Confirmed clean but still quarantined: "
            f"{', '.join(sorted(to_lift))} -> LIFT one."
        )

    if not to_quarantine and not to_lift and not uninspected_nodes(observation) and pending_text != "none":
        hints.append("All nodes inspected or pending. Await lab results -> WAIT.")

    hint_block = "\n".join(hints) if hints else "No urgent action hints — use your judgment."

    return (
        f"=== STEP {step} OBSERVATION ===\n\n"
        f"Natural language summary:\n{observation.natural_language_summary}\n\n"
        f"=== ACTION HINTS ===\n{hint_block}\n\n"
        "=== STRUCTURED FIELDS ===\n"
        f"- lab_results: {json.dumps(observation.lab_results, sort_keys=True, separators=(',', ':'))}\n"
        f"- quarantine_status: {json.dumps(observation.quarantine_status, sort_keys=True, separators=(',', ':'))}\n"
        f"- sensor_readings: {json.dumps(observation.sensor_readings, sort_keys=True, separators=(',', ':'))}\n"
        f"- pending_inspections: {pending_text}\n"
        f"- already_inspected_or_pending_nodes: {inspected_pending_text}\n"
        f"- nodes_available_to_inspect: {uninspected_text}\n"
        f"- illness_reports: {json.dumps(illness_payload, sort_keys=True, separators=(',', ':'))}\n"
        f"- lab_budget: {observation.lab_budget}\n"
        f"- recall_budget: {observation.recall_budget}\n"
        f"- public_trust: {observation.public_trust:.2f}\n"
        f"- visible_batch_ids: {batch_ids}\n"
        f"- available_node_ids: {node_ids}\n"
        f"- recent_history:\n{recent_history}\n\n"
        "Output exactly one valid action."
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


def candidate_texts_from_response(response_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str | None) -> None:
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


def parse_candidate_action(candidate: str, observation: FoodCrisisObservation) -> str | None:
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
    batch_lookup = set(visible_batch_ids(observation))

    if verb in {"INSPECT", "QUARANTINE", "LIFT"}:
        return f"{verb} {normalized_target}" if normalized_target in node_lookup else None
    if verb == "ALERT":
        node = node_lookup.get(normalized_target)
        return f"ALERT {normalized_target}" if (node and node.node_type == "retailer") else None
    if verb in {"RECALL", "TRACE"}:
        return f"{verb} {normalized_target}" if normalized_target in batch_lookup else None

    return None


def parse_model_action(response_text: str, observation: FoodCrisisObservation) -> str | None:
    for candidate in candidate_texts_from_response(response_text):
        parsed = parse_candidate_action(candidate, observation)
        if parsed:
            return parsed
    return None


def apply_action_guard(action_str: str, observation: FoodCrisisObservation) -> str:
    """Block actions that would be wasted or cause errors in the environment."""
    parts = action_str.split(" ", 1)
    verb = parts[0].upper()
    target = parts[1].strip().lower() if len(parts) > 1 else ""

    pending_nodes = set(extract_pending_inspection_nodes(observation))
    known_results = {nid.lower(): result for nid, result in observation.lab_results.items()}

    if verb == "INSPECT":
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

    return action_str


def request_model_completion(client: Any, model_name: str, messages: list[dict[str, str]]) -> str:
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
    history: list[str],
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

        # Retry once on invalid response
        time.sleep(LLM_RATE_LIMIT_SLEEP)
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


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = normalize_single_line(error) if error else "null"
    done_val = str(bool(done)).lower()
    print(
        f'[STEP] step={step} action="{normalize_single_line(action)}" reward={reward:.2f} done={done_val} error={error_val}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def extract_step_error(observation: FoodCrisisObservation) -> str | None:
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
            action_str = apply_action_guard(action_str, observation)

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