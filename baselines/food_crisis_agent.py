from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("FOODCRISIS_STANDALONE", "1")

from irce.environment import FoodCrisisEnv
from irce.grading import grade_episode
from irce.models import FoodCrisisAction, FoodCrisisObservation, NodeState
from irce.tasks import TASKS

NODE_PRIORITY = {"farm": 0, "processing": 1, "warehouse": 2, "retailer": 3}


@dataclass
class BaselineMemory:
    pending_inspections: set[str] = field(default_factory=set)
    recalled_batches: set[str] = field(default_factory=set)
    alerted_retailers: set[str] = field(default_factory=set)

    def sync(self, observation: FoodCrisisObservation) -> None:
        self.pending_inspections.difference_update(observation.lab_results.keys())


class FoodCrisisBaselineAgent:
    """Deterministic baseline using exact FoodCrisis actions."""

    def __init__(self) -> None:
        self.memory = BaselineMemory()

    def reset(self) -> None:
        self.memory = BaselineMemory()

    def act(self, observation: FoodCrisisObservation) -> str:
        self.memory.sync(observation)
        nodes = {node.node_id: node for node in observation.nodes}

        lift_target = self._choose_lift_target(observation)
        if lift_target:
            return f"LIFT {lift_target}"

        quarantine_target = self._choose_quarantine_target(observation)
        if quarantine_target:
            return f"QUARANTINE {quarantine_target}"

        recall_batch = self._choose_recall_batch(observation, nodes)
        if recall_batch:
            self.memory.recalled_batches.add(recall_batch)
            return f"RECALL {recall_batch}"

        inspect_target = self._choose_inspection_target(observation, nodes)
        if inspect_target:
            self.memory.pending_inspections.add(inspect_target)
            return f"INSPECT {inspect_target}"

        alert_target = self._choose_alert_target(observation, nodes)
        if alert_target:
            self.memory.alerted_retailers.add(alert_target)
            return f"ALERT {alert_target}"

        return "WAIT"

    def _choose_lift_target(self, observation: FoodCrisisObservation) -> str | None:
        candidates = [
            node_id
            for node_id, result in observation.lab_results.items()
            if result == "clean" and observation.quarantine_status.get(node_id, False)
        ]
        return sorted(candidates)[0] if candidates else None

    def _choose_quarantine_target(self, observation: FoodCrisisObservation) -> str | None:
        contaminated_nodes = [
            node_id
            for node_id, result in observation.lab_results.items()
            if result == "contaminated" and not observation.quarantine_status.get(node_id, False)
        ]
        if not contaminated_nodes:
            return None

        return sorted(
            contaminated_nodes,
            key=lambda node_id: (NODE_PRIORITY.get(self._node_type(observation, node_id), 99), node_id),
        )[0]

    def _choose_recall_batch(
        self,
        observation: FoodCrisisObservation,
        nodes: dict[str, NodeState],
    ) -> str | None:
        if observation.recall_budget < 10:
            return None

        reported_retailers = sorted(report.retailer_id for report in observation.illness_reports if report.retailer_id in nodes)
        contaminated_nodes = sorted(
            (node_id for node_id, result in observation.lab_results.items() if result == "contaminated" and node_id in nodes),
            key=lambda node_id: (NODE_PRIORITY.get(nodes[node_id].node_type, 99), node_id),
        )

        search_order = reported_retailers + contaminated_nodes
        seen_nodes: set[str] = set()
        for node_id in search_order:
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            for batch_id in sorted(nodes[node_id].batch_ids):
                if batch_id not in self.memory.recalled_batches:
                    return batch_id
        return None

    def _choose_inspection_target(
        self,
        observation: FoodCrisisObservation,
        nodes: dict[str, NodeState],
    ) -> str | None:
        if observation.lab_budget <= 0:
            return None

        urgent_retailers = {report.retailer_id for report in observation.illness_reports}
        contaminated_nodes = {node_id for node_id, result in observation.lab_results.items() if result == "contaminated"}

        def candidate_score(node: NodeState) -> tuple[float, int, str]:
            report_bonus = 0.15 if node.node_id in urgent_retailers else 0.0
            quarantine_bonus = 0.1 if node.node_id in contaminated_nodes else 0.0
            priority_bonus = {0: 0.08, 1: 0.05, 2: 0.02, 3: 0.0}.get(NODE_PRIORITY.get(node.node_type, 99), 0.0)
            return (-(node.sensor_reading + report_bonus + quarantine_bonus + priority_bonus), NODE_PRIORITY.get(node.node_type, 99), node.node_id)

        candidates = [
            node
            for node in nodes.values()
            if node.node_id not in observation.lab_results
            and node.node_id not in self.memory.pending_inspections
            and not observation.quarantine_status.get(node.node_id, False)
        ]
        if not candidates:
            return None

        best = sorted(candidates, key=candidate_score)[0]
        if best.sensor_reading >= 0.45 or not observation.lab_results:
            return best.node_id
        return None

    def _choose_alert_target(
        self,
        observation: FoodCrisisObservation,
        nodes: dict[str, NodeState],
    ) -> str | None:
        if observation.public_trust <= 0.35:
            return None

        retailers = [
            node
            for node in nodes.values()
            if node.node_type == "retailer"
            and node.node_id not in self.memory.alerted_retailers
            and not observation.quarantine_status.get(node.node_id, False)
        ]
        if not retailers:
            return None

        risky = sorted(retailers, key=lambda node: (-node.sensor_reading, node.node_id))
        target = risky[0]
        if target.sensor_reading >= 0.75 or any(report.retailer_id == target.node_id for report in observation.illness_reports):
            return target.node_id
        return None

    def _node_type(self, observation: FoodCrisisObservation, node_id: str) -> str:
        for node in observation.nodes:
            if node.node_id == node_id:
                return node.node_type
        return "unknown"


def run_episode(task_id: int, seed: int) -> float:
    env = FoodCrisisEnv(task_id=task_id, seed=seed)
    agent = FoodCrisisBaselineAgent()
    observation = env.reset(seed=seed, task_id=task_id)

    while not observation.done:
        action = agent.act(observation)
        observation = env.step(FoodCrisisAction(action_type=action))

    return grade_episode(env.episode_log)


def main() -> None:
    scores: list[float] = []
    for task_id in sorted(TASKS):
        score = run_episode(task_id=task_id, seed=7)
        scores.append(score)
        print(f"task_{task_id} score: {score:.3f}")

    average = sum(scores) / len(scores) if scores else 0.0
    print(f"average score: {average:.3f}")


if __name__ == "__main__":
    main()
