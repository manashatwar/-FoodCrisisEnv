#!/usr/bin/env python3
"""
train_grpo.py
=============
True GRPO (Group Relative Policy Optimization) training for FoodCrisisEnv.

A real language model is used as the policy. It generates action text at each
environment step. The final episode score from grading.py is the reward.

GRPO advantage  = episode_score_k - mean(group_scores)
GRPO loss       = -advantage * mean(log_pi(action_tokens | prompt))

Modes
-----
  manual  : Manual GRPO loop using HF model.generate(). CPU-compatible (slow).
  trl     : TRL GRPOTrainer. GPU recommended.

Quick start
-----------
  # Smallest model, CPU test (no GPU needed)
  python train_grpo.py --mode manual --model Qwen/Qwen2.5-0.5B-Instruct --episodes 9 --group-size 3

  # Memory-efficient GPU run with LoRA
  python train_grpo.py --mode manual --model Qwen/Qwen2.5-1.5B-Instruct --lora --device cuda

  # Full TRL GRPOTrainer (GPU required)
  python train_grpo.py --mode trl --model Qwen/Qwen2.5-1.5B-Instruct --lora

  # Use saved model in inference.py:
  #   Set MODEL_NAME=./grpo_trained_model and API_BASE_URL= in .env
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Standalone flag: keeps environment independent of openenv runtime
os.environ.setdefault("FOODCRISIS_STANDALONE", "1")

from irce.environment import FoodCrisisEnv
from irce.grading import grade_episode
from irce.models import FoodCrisisAction, FoodCrisisObservation
from irce.tasks import build_task_registry, get_task_config
from inference import (
    SYSTEM_PROMPT as INFERENCE_SYSTEM_PROMPT,
    build_user_prompt,
    extract_raw_action_line,
)

SUPPORTED_ACTIONS = ("INSPECT", "QUARANTINE", "LIFT", "RECALL", "TRACE", "ALERT", "WAIT")
TRAIN_MAX_LENGTH = int(os.getenv("GRPO_MAX_LENGTH", "512"))


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ── prompt templates (same format as inference.py for compatibility) ───────────

SYSTEM_PROMPT = (
    "You are a legacy-compatible policy controller for FoodCrisisEnv.\n"
    "Choose the single best recovery action based on the current error_type and context.\n\n"
    "Decision rules (follow in order):\n"
    "  1. error=HARD        → MODIFY  (never RETRY on HARD — it always fails)\n"
    "  2. error=RATE_LIMIT  → SWITCH  (cooldown — switching tool path clears it)\n"
    "  3. repeat_errors>=2  → SWITCH  (same error twice means path is stuck)\n"
    "  4. result=AMBIGUOUS  → REPLAN  (partial signal — reframe before next attempt)\n"
    "  5. error=TRANSIENT   → RETRY   (transient failures are safe to retry once)\n"
    "  6. budget<0.15       → ESCALATE (too little left to keep trying)\n\n"
    "Actions:\n"
    "  RETRY    — safe only when error is TRANSIENT and no cooldown\n"
    "  MODIFY   — best first response to HARD errors\n"
    "  SWITCH   — escapes RATE_LIMIT and repeated-failure loops\n"
    "  REPLAN   — resolves AMBIGUOUS outcomes and cooldown pressure\n"
    "  ESCALATE — absolute last resort only\n\n"
    "Respond with EXACTLY one word: RETRY, MODIFY, SWITCH, REPLAN, or ESCALATE\n"
    "No explanation. One word only."
)


SYSTEM_PROMPT = INFERENCE_SYSTEM_PROMPT

def build_step_prompt(step: int, obs: FoodCrisisObservation, history: list[str]) -> str:
    """Build the user-turn prompt for one environment step. Matches inference.py format."""
    return build_user_prompt(step, obs, history)


def parse_action(text: str) -> str:
    """Extract a valid action from LLM output, defaulting to REPLAN."""
    m = re.search(r"\b(RETRY|MODIFY|SWITCH|REPLAN|ESCALATE)\b", text.upper())
    return m.group(1) if m else "REPLAN"


def parse_and_guard_action(
    completion_text: str,
    observation: FoodCrisisObservation,
    history: list[str],
) -> str:
    """Pure-training path: use the first non-empty output line exactly as pure inference does."""
    return extract_raw_action_line(completion_text)


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class StepData:
    """Token-level record from one LLM call within an episode."""
    prompt_text: str
    completion_text: str
    action: str


@dataclass
class EpisodeData:
    """Full record from one FoodCrisisEnv episode driven by the LLM policy."""
    task_id: int
    seed: int
    steps: list[StepData] = field(default_factory=list)
    score: float = 0.0
    completed: bool = False


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode_with_llm(
    model: Any,
    tokenizer: Any,
    task_id: int,
    seed: int,
    device: str,
    temperature: float = 0.9,
    max_new_tokens: int = 12,
    deception_level: float = 0.0,
) -> EpisodeData:
    """
    Run one full FoodCrisisEnv episode using model.generate() as the policy.

    The LLM is called once per environment step. All (prompt, completion) pairs
    are stored for the GRPO loss computation after the episode ends.
    """
    import torch

    env = FoodCrisisEnv(task_id=task_id, seed=seed)
    task_config = get_task_config(task_id)
    observation = env.reset(seed=seed, task_id=task_id, deception_level=getattr(env, '_active_deception_level', 0.0))

    episode = EpisodeData(task_id=task_id, seed=seed)
    history: list[str] = []

    for step in range(1, task_config.max_steps + 1):
        if observation.done:
            break

        # Build chat prompt
        user_content = build_user_prompt(step, observation, history)
        messages = [
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt_text: str = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            prompt_text = f"{SYSTEM_PROMPT}\n\nUser: {user_content}\nAssistant:"

        # Tokenise and generate (no grad — we recompute for backward pass)
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=TRAIN_MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0, prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        action = parse_and_guard_action(completion_text, observation, history)

        episode.steps.append(
            StepData(
                prompt_text=prompt_text,
                completion_text=action,
                action=action,
            )
        )

        observation = env.step(FoodCrisisAction(action_type=action))
        history.append(
            f"step={step} action={action} result={observation.tool_result} "
            f"reward={float(observation.reward or 0.0):+.3f}"
        )

    episode.score = grade_episode(env.episode_log)
    episode.completed = bool(env.episode_log and env.episode_log[-1].get("completed"))
    return episode


# ── GRPO loss ─────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    model: Any,
    tokenizer: Any,
    group_episodes: list[EpisodeData],
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """
    GRPO loss for one group of K episodes.

    For each episode k:
        advantage_k = (score_k - mean_scores) / std_scores   [normalised]

    For each (prompt, completion) step inside episode k:
        step_loss = -advantage_k * mean(log_pi(completion_tokens | prompt))

    Final loss = mean over all steps across all K episodes.
    This is then backpropagated to update the model weights.
    """
    import torch
    import torch.nn.functional as F

    scores = [ep.score for ep in group_episodes]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = max(1e-8, variance ** 0.5)
    advantages = [(s - mean_score) / std_score for s in scores]

    valid_step_specs: list[tuple[StepData, float, int]] = []

    for episode, advantage in zip(group_episodes, advantages):
        for step in episode.steps:
            prompt_ids_only = tokenizer(
                step.prompt_text, return_tensors="pt", truncation=True, max_length=TRAIN_MAX_LENGTH
            )
            prompt_len = prompt_ids_only["input_ids"].shape[1]

            full_inputs = tokenizer(
                step.prompt_text + step.completion_text,
                return_tensors="pt",
                truncation=True,
                max_length=TRAIN_MAX_LENGTH,
            )
            total_len = full_inputs["input_ids"].shape[1]
            comp_start = min(prompt_len - 1, total_len - 2)
            comp_end = total_len - 1

            if comp_start >= comp_end:
                continue
            valid_step_specs.append((step, float(advantage), int(prompt_len)))

    metrics: dict[str, Any] = {
        "mean_score": mean_score,
        "scores": scores,
        "advantages": [round(a, 4) for a in advantages],
    }

    if not valid_step_specs:
        # No valid completions — return a zero loss attached to params so backward still works
        dummy_param = next(p for p in model.parameters() if p.requires_grad)
        zero_loss = dummy_param.sum() * 0.0
        metrics["loss"] = 0.0
        return zero_loss, metrics

    total_loss_value = 0.0
    step_count = len(valid_step_specs)

    for step, advantage, prompt_len in valid_step_specs:
        full_inputs = tokenizer(
            step.prompt_text + step.completion_text,
            return_tensors="pt",
            truncation=True,
            max_length=TRAIN_MAX_LENGTH,
        ).to(device)

        total_len = full_inputs["input_ids"].shape[1]
        comp_start = min(prompt_len - 1, total_len - 2)
        comp_end = total_len - 1
        if comp_start >= comp_end:
            continue

        outputs = model(**full_inputs, use_cache=False)
        logits = outputs.logits[0]
        completion_logits = logits[comp_start:comp_end]
        completion_ids = full_inputs["input_ids"][0, comp_start + 1 : comp_end + 1]

        if len(completion_ids) == 0:
            continue

        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs[range(len(completion_ids)), completion_ids]
        step_loss = -advantage * token_log_probs.mean()
        total_loss_value += float(step_loss.detach().item())
        (step_loss / step_count).backward()

        del outputs, logits, completion_logits, completion_ids, log_probs, token_log_probs, step_loss, full_inputs
        if str(device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    mean_loss = total_loss_value / max(1, step_count)
    metrics["loss"] = mean_loss
    dummy_param = next(p for p in model.parameters() if p.requires_grad)
    return dummy_param.new_tensor(mean_loss), metrics


def push_output_to_hub(local_dir: Path, repo_id: str) -> None:
    """Upload the saved model/adapters to a Hugging Face model repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --push-to-hub-repo. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    token = (os.getenv("HF_TOKEN") or "").strip() or None
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload GRPO training output",
    )
    print(f"  Uploaded to HF Hub: https://huggingface.co/{repo_id}")


# ── manual GRPO training loop ─────────────────────────────────────────────────

def train_manual_grpo(
    model_name: str,
    episodes: int,
    group_size: int,
    learning_rate: float,
    temperature: float,
    seed: int,
    use_lora: bool,
    save_path: Path,
    device_str: str,
    push_to_hub_repo: str | None,
    deception_level: float = 0.0,
    load_in_4bit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> None:
    """
    Manual GRPO training loop.

    For each training step:
      1. Pick a task (cycling through 1, 2, 3)
      2. Run K = group_size episodes in parallel (different seeds, same task)
      3. Each episode: LLM generates one action per env step
      4. Compute group-relative advantages from final scores
      5. Recompute log_probs with gradient and backprop GRPO loss
      6. Adam step

    Works on CPU (slow) or GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("FoodCrisisEnv  -  Manual GRPO Training")
    print(SEP)
    print(f"  Model      : {model_name}")
    print(f"  Device     : {device_str}")
    print(f"  Episodes   : {episodes}")
    print(f"  Group size : {group_size}  (K completions per GRPO step)")
    print(f"  LR         : {learning_rate}")
    print(f"  LoRA       : {use_lora}")
    print(f"  4-bit load : {load_in_4bit}")
    print(f"  Save path  : {save_path}")
    print(f"{SEP}\n")

    device = torch.device(device_str)

    # ── load tokenizer ─────────────────────────────────────────────────────────
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── load model ─────────────────────────────────────────────────────────────
    print("Loading model …")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device_str == "cpu":
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["low_cpu_mem_usage"] = True
    elif load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "4-bit training requires bitsandbytes support in transformers. "
                "Install with: pip install -U bitsandbytes transformers accelerate peft"
            ) from exc

        cuda_index = 0
        if ":" in device_str:
            try:
                cuda_index = int(device_str.split(":", 1)[1])
            except ValueError:
                cuda_index = 0
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": cuda_index}
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if not load_in_4bit:
        model = model.to(device)
    if hasattr(model, "config"):
        model.config.use_cache = False

    # ── optional LoRA ──────────────────────────────────────────────────────────
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            if load_in_4bit:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            if hasattr(model, "config"):
                model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.print_trainable_parameters()
        except ImportError:
            print("⚠  peft not installed — LoRA skipped. Install: pip install peft")

    # ── optimizer ──────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if load_in_4bit and device_str.startswith("cuda"):
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=learning_rate)
            print("  Optimizer  : bitsandbytes PagedAdamW8bit")
        except Exception as exc:
            print(f"  Optimizer  : torch Adam (bitsandbytes optimizer unavailable: {exc})")
            optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    task_registry = build_task_registry()
    task_ids = sorted(task_registry)
    all_scores: list[float] = []
    training_log: list[dict[str, Any]] = []
    best_avg = 0.0
    episode_count = 0
    step_num = 0

    print("Starting GRPO training …\n")

    # Dynamic curriculum: starts at the requested deception_level and increases when agent improves
    current_deception_level = float(deception_level)

    while episode_count < episodes:
        for task_id in task_ids:
            if episode_count >= episodes:
                break

            step_num += 1
            t0 = time.time()

            print(f"{'─'*55}")
            print(
                f"Step {step_num:03d} | Task {task_id} "
                f"({task_registry[task_id].name}) | "
                f"Collecting {group_size} rollouts …"
            )

            # ── rollout phase (no grad) ────────────────────────────────────────
            group_episodes: list[EpisodeData] = []
            model.eval()

            for k in range(group_size):
                ep_seed = seed + (episode_count * 31) + (k * 7) + task_id
                ep = run_episode_with_llm(
                    model=model,
                    tokenizer=tokenizer,
                    task_id=task_id,
                    seed=ep_seed,
                    device=device_str,
                    temperature=temperature,
                    deception_level=current_deception_level,
                )
                group_episodes.append(ep)
                episode_count += 1
                all_scores.append(ep.score)

                actions_taken = [s.action for s in ep.steps]
                print(
                    f"  k={k + 1}: seed={ep_seed}  steps={len(ep.steps)}"
                    f"  actions={actions_taken}"
                    f"  score={ep.score:.3f}"
                    f"  {'✓ done' if ep.completed else '✗'}"
                )

            scores = [ep.score for ep in group_episodes]
            mean_sc = sum(scores) / len(scores)
            adv_list = [round(s - mean_sc, 4) for s in scores]

            print(f"\n  Scores     : {[f'{s:.3f}' for s in scores]}")
            print(f"  Mean score : {mean_sc:.3f}")
            print(f"  Advantages : {adv_list}")
            print(f"  Deception  : {current_deception_level:.3f}")
            # Dynamic curriculum: increase deception when doing well
            if mean_sc > 0.65 and current_deception_level < 1.0:
                current_deception_level = round(min(1.0, current_deception_level + 0.05), 3)
                print(f"  Deception up -> {current_deception_level:.3f}")

            # ── update phase (with grad) ───────────────────────────────────────
            if device_str.startswith("cuda"):
                torch.cuda.empty_cache()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                group_episodes=group_episodes,
                device=device_str,
            )
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            elapsed = time.time() - t0
            recent_avg = sum(all_scores[-10:]) / min(len(all_scores), 10)

            print(f"\n  Loss       : {metrics['loss']:.5f}")
            print(f"  Recent avg : {recent_avg:.3f}  (last 10 episodes)")
            print(f"  Elapsed    : {elapsed:.1f}s")

            if mean_sc > best_avg:
                best_avg = mean_sc
                print(f"  ★ New best group avg: {best_avg:.3f}")

            training_log.append(
                {
                    "step": step_num,
                    "task_id": task_id,
                    "scores": scores,
                    "mean_score": mean_sc,
                    "loss": metrics["loss"],
                    "episode_total": episode_count,
                }
            )

    # ── save ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Training complete.")
    print(f"  Overall avg score : {sum(all_scores)/len(all_scores):.3f}")
    print(f"  Best group avg    : {best_avg:.3f}")

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  Model saved to    : {save_path}")

    log_path = save_path / "grpo_training_log.json"
    log_path.write_text(json.dumps(training_log, indent=2), encoding="utf-8")
    print(f"  Log saved to      : {log_path}")

    if push_to_hub_repo:
        print(f"  Pushing to HF Hub : {push_to_hub_repo}")
        push_output_to_hub(save_path, push_to_hub_repo)

    print("\nTo run inference with this model:")
    print(f"  Set MODEL_NAME={save_path} in .env")
    print(f"  Leave API_BASE_URL empty (local HF model)\n")


# ── TRL GRPOTrainer integration ───────────────────────────────────────────────

def train_trl_grpo(
    model_name: str,
    episodes: int,
    group_size: int,
    learning_rate: float,
    temperature: float,
    seed: int,
    use_lora: bool,
    save_path: Path,
    push_to_hub_repo: str | None,
) -> None:
    """
    TRL GRPOTrainer-based training.

    The dataset is a list of (task_id, seed) metadata strings.
    The rollout_func runs a full FoodCrisisEnv episode for each dataset prompt,
    concatenates all (step_prompt, action) token sequences, and returns
    the episode score as the reward for the GRPOTrainer.

    Requirements
    ------------
      pip install trl>=0.12.0 datasets transformers peft
      GPU with ≥ 8 GB VRAM (16 GB+ recommended for 1.5B models)
    """
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        print(f"\n❌ TRL mode requires additional dependencies: {exc}")
        print("   Install: pip install trl>=0.12.0 datasets transformers peft")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("FoodCrisisEnv  -  TRL GRPOTrainer")
    print(f"  Model  : {model_name}  |  Device: {device}")
    print(f"  K      : {group_size}  |  LR: {learning_rate}")
    if device == "cpu":
        print("  ⚠  CPU detected — TRL GRPO will be very slow. Use --mode manual.")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset: each row encodes task_id + seed as a metadata prompt string.
    task_registry = build_task_registry()
    entries: list[dict[str, Any]] = []
    for i in range(episodes * group_size):
        task_id = (i % len(task_registry)) + 1
        ep_seed = seed + i * 13
        entries.append({"prompt": f"FoodCrisisEnv|task_id={task_id}|seed={ep_seed}"})
    train_dataset = Dataset.from_list(entries)

    def rollout_func(prompts: list[str], trainer: Any = None) -> dict[str, Any]:
        """
        TRL rollout function.

        Runs one full FoodCrisisEnv episode per prompt, concatenating all step
        (prompt_ids, completion_ids, logprobs) across steps. The episode
        score is passed back as the reward via the dataset column.
        """
        _model = trainer.model if trainer is not None else None
        if _model is None:
            raise RuntimeError("rollout_func requires a trainer with a loaded model.")

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_scores: list[float] = []

        import torch
        import torch.nn.functional as F

        for meta in prompts:
            try:
                task_id = int(re.search(r"task_id=(\d+)", meta).group(1))  # type: ignore[union-attr]
                ep_seed = int(re.search(r"seed=(\d+)", meta).group(1))  # type: ignore[union-attr]
            except (AttributeError, ValueError):
                task_id, ep_seed = 1, seed

            task_config = get_task_config(task_id)
            env = FoodCrisisEnv(task_id=task_id, seed=ep_seed)
            observation = env.reset(seed=ep_seed, task_id=task_id)

            ep_prompt_ids: list[int] = []
            ep_comp_ids: list[int] = []
            ep_logprobs: list[float] = []
            history: list[str] = []

            for step in range(1, task_config.max_steps + 1):
                if observation.done:
                    break

                user_content = build_user_prompt(step, observation, history)
                messages = [
                    {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                try:
                    prompt_text: str = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                except Exception:
                    prompt_text = f"{SYSTEM_PROMPT}\n\nUser: {user_content}\nAssistant:"

                tok_in = tokenizer(
                    prompt_text, return_tensors="pt", truncation=True, max_length=TRAIN_MAX_LENGTH
                ).to(device)
                input_len = tok_in["input_ids"].shape[1]

                with torch.no_grad():
                    gen_out = _model.generate(
                        **tok_in,
                        max_new_tokens=12,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                comp_ids = gen_out.sequences[0, input_len:].tolist()
                # Per-token log probs from model scores (one score tensor per generated token)
                step_lps: list[float] = [
                    float(F.log_softmax(score[0], dim=-1)[tok_id].item())
                    for tok_id, score in zip(comp_ids, gen_out.scores)
                ]

                ep_prompt_ids.extend(tok_in["input_ids"][0].tolist())
                ep_comp_ids.extend(comp_ids)
                ep_logprobs.extend(step_lps)

                comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                action = parse_and_guard_action(comp_text, observation, history)

                observation = env.step(FoodCrisisAction(action_type=action))
                history.append(f"step={step} action={action} result={observation.tool_result}")

            all_prompt_ids.append(ep_prompt_ids)
            all_completion_ids.append(ep_comp_ids)
            all_logprobs.append(ep_logprobs)
            all_scores.append(grade_episode(env.episode_log))

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "irce_score": all_scores,
        }

    def reward_irce_score(completions: list[str], **kwargs: Any) -> list[float]:
        """Reward function: the FoodCrisisEnv grader score for each episode."""
        return [float(s) for s in kwargs.get("irce_score", [0.0] * len(completions))]

    # ── LoRA config ────────────────────────────────────────────────────────────
    peft_config = None
    if use_lora:
        try:
            from peft import LoraConfig, TaskType

            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        except ImportError:
            print("⚠  peft not installed — LoRA skipped.")

    grpo_cfg = GRPOConfig(
        output_dir=str(save_path),
        num_train_epochs=1,
        per_device_train_batch_size=group_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_generations=group_size,
        max_completion_length=16,
        max_prompt_length=512,
        logging_steps=1,
        save_steps=10,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=(torch.cuda.is_available()),
    )

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[reward_irce_score],
        train_dataset=train_dataset,
        args=grpo_cfg,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    print("Starting TRL GRPO training …")
    trainer.train()

    print(f"\nSaving to {save_path} …")
    trainer.save_model(str(save_path))
    if push_to_hub_repo:
        print(f"Uploading to HF Hub: {push_to_hub_repo}")
        push_output_to_hub(save_path, push_to_hub_repo)
    print("✅ TRL GRPO training complete!")


# ── CLI ───────────────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> str:
    if device_str != "auto":
        return device_str
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def model_size_billion_hint(model_name: str) -> float | None:
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b(?:\b|-|_)", model_name, re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))


def validate_training_device(args: argparse.Namespace, device: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Install training dependencies first.") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but PyTorch cannot use CUDA.\n"
            "In Colab: Runtime > Change runtime type > T4 GPU, then restart runtime.\n"
            "In Kaggle: Settings > Accelerator > GPU, then restart the session.\n"
            "Then verify with: import torch; print(torch.cuda.is_available())."
        )

    if device.startswith("cuda"):
        size_b = model_size_billion_hint(args.model)
        allow_full = env_flag("ALLOW_FULL_PRECISION_7B", False)
        if size_b is not None and size_b >= 7.0 and not args.load_in_4bit and not allow_full:
            raise SystemExit(
                f"{args.model} looks like a {size_b:g}B model. Full fp16 GRPO is likely to OOM "
                "on a 16GB Kaggle/Colab GPU.\n"
                "Rerun with --load-in-4bit for QLoRA, or set ALLOW_FULL_PRECISION_7B=1 "
                "if you really want to risk full-precision training."
            )
        return

    allow_cpu = os.getenv("ALLOW_CPU_TRAINING", "").strip().lower() in {"1", "true", "yes", "on"}
    size_b = model_size_billion_hint(args.model)
    if allow_cpu or size_b is None or size_b < 1.0:
        return

    raise SystemExit(
        f"Training resolved to CPU, but {args.model} looks like a {size_b:g}B model.\n"
        "That will be extremely slow and may appear stuck while loading weights.\n\n"
        "Fix in Colab:\n"
        "  1. Runtime > Change runtime type > T4 GPU\n"
        "  2. Restart runtime\n"
        "  3. Run: import torch; print(torch.cuda.is_available())\n"
        "  4. Rerun training with --device cuda\n\n"
        "For CPU testing, use Qwen/Qwen2.5-0.5B-Instruct or set ALLOW_CPU_TRAINING=1."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="True GRPO LLM training for the FoodCrisisEnv benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CPU-compatible quick test (smallest model)
  python train_grpo.py --mode manual --model Qwen/Qwen2.5-0.5B-Instruct --episodes 9 --group-size 3

  # Memory-efficient GPU run with LoRA adapter
  python train_grpo.py --mode manual --model Qwen/Qwen2.5-1.5B-Instruct --lora --device cuda

  # Full TRL GRPOTrainer (requires GPU + pip install trl datasets)
  python train_grpo.py --mode trl --model Qwen/Qwen2.5-1.5B-Instruct --lora --episodes 50
        """,
    )
    p.add_argument(
        "--mode",
        choices=["manual", "trl"],
        default="manual",
        help="manual = pure PyTorch GRPO loop (CPU-ok); trl = TRL GRPOTrainer (GPU recommended)",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or local path to a saved model/adapter",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Total training episodes (each is one full LLM-driven env run)",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="K: rollouts per task per GRPO step (advantage computed within group)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Adam learning rate",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Generation temperature (higher = more exploration diversity)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for reproducibility",
    )
    p.add_argument(
        "--lora",
        action="store_true",
        help="Apply LoRA (PEFT) for memory-efficient training (requires: pip install peft)",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=env_flag("LOAD_IN_4BIT", False),
        help="Load the base model in 4-bit NF4 for QLoRA-style training (recommended for 7B on 16GB GPUs)",
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=int(os.getenv("LORA_R", "8")),
        help="LoRA rank. Use 4 for tighter memory, 8 for the default.",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=int(os.getenv("LORA_ALPHA", "16")),
        help="LoRA alpha. Commonly 2x lora-r.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device override: auto | cpu | cuda | cuda:0 | mps",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=ROOT / "grpo_trained_model",
        help="Directory for saved model or LoRA adapter",
    )
    p.add_argument(
        "--deception-level",
        type=float,
        default=0.0,
        help="Starting deception level (0.0=none, 1.0=max). Dynamic curriculum auto-increases.",
    )
    p.add_argument(
        "--push-to-hub-repo",
        default=None,
        help="Optional Hugging Face model repo id to upload the saved training output",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    validate_training_device(args, device)

    if args.mode == "manual":
        train_manual_grpo(
            model_name=args.model,
            episodes=args.episodes,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            seed=args.seed,
            use_lora=args.lora,
            save_path=args.save_path,
            device_str=device,
            push_to_hub_repo=args.push_to_hub_repo,
            deception_level=getattr(args, 'deception_level', 0.0),
            load_in_4bit=args.load_in_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
    else:
        train_trl_grpo(
            model_name=args.model,
            episodes=args.episodes,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            seed=args.seed,
            use_lora=args.lora,
            save_path=args.save_path,
            push_to_hub_repo=args.push_to_hub_repo,
        )


if __name__ == "__main__":
    main()
