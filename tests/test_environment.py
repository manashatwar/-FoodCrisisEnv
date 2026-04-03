from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("IRCE_STANDALONE", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irce.environment import IRCEEnv
from irce.grading import grade_episode
from irce.models import BatchRecord, IRCEAction


def test_reset_returns_observation() -> None:
    env = IRCEEnv(task_id=1, seed=7)
    observation = env.reset(seed=7, task_id=1)
    assert observation.timestep == 0
    assert observation.sensor_readings
    assert observation.lab_budget == 10


def test_quarantine_blocks_spread() -> None:
    env = IRCEEnv(task_id=1, seed=7)
    env.reset(seed=7, task_id=1)
    source = env.state.source_nodes[0]
    downstream = env.state.nodes[source].connected_to[0]
    before = env.state.true_contamination[downstream]
    env.step(IRCEAction(action_type=f"QUARANTINE {source}"))
    after = env.state.true_contamination[downstream]
    assert after == before


def test_inspect_costs_budget_and_returns_result_next_step() -> None:
    env = IRCEEnv(task_id=1, seed=7)
    env.reset(seed=7, task_id=1)
    target = env.state.source_nodes[0]
    env.step(IRCEAction(action_type=f"INSPECT {target}"))
    assert env.state.lab_budget == 9
    assert target not in env.state.lab_results
    env.step(IRCEAction(action_type="WAIT"))
    assert env.state.lab_results[target] in {"clean", "contaminated"}


def test_recall_removes_batch() -> None:
    env = IRCEEnv(task_id=1, seed=7)
    env.reset(seed=7, task_id=1)
    batch_id = next(batch.batch_id for batch in env.state.batch_records.values() if batch.contaminated)
    current_node = env.state.batch_records[batch_id].current_node
    env.step(IRCEAction(action_type=f"RECALL {batch_id}"))
    assert env.state.batch_records[batch_id].recalled is True
    assert batch_id not in env.state.nodes[current_node].batch_ids


def test_illness_report_delay() -> None:
    env = IRCEEnv(task_id=2, seed=7)
    env.reset(seed=7, task_id=2)
    retailer = env.task_config.retailer_ids[0]
    batch_id = "forced_batch"
    env.state.batch_records[batch_id] = BatchRecord(
        batch_id=batch_id,
        origin_node=env.state.source_nodes[0],
        current_node=retailer,
        path_taken=[env.state.source_nodes[0], retailer],
        planned_path=[env.state.source_nodes[0], retailer],
        contaminated=True,
    )
    env.state.nodes[retailer].batch_ids.append(batch_id)
    env.state.total_contaminated_batches += 1
    env.state.true_contamination[retailer] = 0.9

    env.step(IRCEAction(action_type="WAIT"))
    assert not env.state.illness_reports
    env.step(IRCEAction(action_type="WAIT"))
    env.step(IRCEAction(action_type="WAIT"))
    env.step(IRCEAction(action_type="WAIT"))
    assert env.state.illness_reports


def test_grader_is_deterministic() -> None:
    env_one = IRCEEnv(task_id=1, seed=7)
    env_two = IRCEEnv(task_id=1, seed=7)
    env_one.reset(seed=7, task_id=1)
    env_two.reset(seed=7, task_id=1)
    actions = ["MODIFY", "SWITCH", "WAIT", "WAIT"]
    for action in actions:
        env_one.step(IRCEAction(action_type=action))
        env_two.step(IRCEAction(action_type=action))
        if env_one.state.timestep >= env_one.task_config.max_steps or env_two.state.timestep >= env_two.task_config.max_steps:
            break
    assert grade_episode(env_one.episode_log) == grade_episode(env_two.episode_log)
