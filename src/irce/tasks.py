from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    node_type: str
    downstream: tuple[str, ...]


@dataclass(frozen=True)
class ScoreWeights:
    containment: float
    precision: float
    speed: float
    trust: float


@dataclass(frozen=True)
class TaskConfig:
    task_id: int
    name: str
    description: str
    goal: str
    node_specs: tuple[NodeSpec, ...]
    n_sources: int
    sensor_noise_std: float
    illness_delay: int
    lab_budget: int
    recall_budget: int
    max_steps: int
    false_signal_count: int
    reseed_interval: int | None
    batches_per_farm: int
    edge_weight_low: float
    edge_weight_high: float
    contamination_threshold: float
    enable_trace: bool
    enable_hold: bool
    score_weights: ScoreWeights

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(spec.node_id for spec in self.node_specs)

    @property
    def farm_ids(self) -> tuple[str, ...]:
        return tuple(spec.node_id for spec in self.node_specs if spec.node_type == "farm")

    @property
    def retailer_ids(self) -> tuple[str, ...]:
        return tuple(spec.node_id for spec in self.node_specs if spec.node_type == "retailer")



# ---------------------------------------------------------------------------
# Deception scaling — independent of task_id graph structure
# ---------------------------------------------------------------------------

# Per-task caps for deception parameters.
# At deception_level=0.0: parameters match the task's base TaskConfig exactly.
# At deception_level=1.0: parameters reach these maximums.
# The graph (nodes, edges, n_sources) is NEVER mutated by deception_level.
DECEPTION_CAPS: dict[int, dict[str, float | int]] = {
    1: {"max_false_signals": 3, "max_noise": 0.25, "max_delay": 5},
    2: {"max_false_signals": 4, "max_noise": 0.25, "max_delay": 6},
    3: {"max_false_signals": 5, "max_noise": 0.30, "max_delay": 8},
}


def compute_deception_params(
    task_id: int,
    deception_level: float,
    base_config: "TaskConfig",
) -> dict[str, object]:
    """Return effective adversarial params for a given (task_id, deception_level) pair.

    Uses simple linear interpolation between the task's base values (at dec=0.0)
    and the per-task cap (at dec=1.0).  Accepts ANY float in [0.0, 1.0].

    The formula is deterministic and instant — no lookup table.

    Example (Task 1, deception_level=0.37):
        false_signal_count = int(0.37 * 3)           = 1
        sensor_noise_std   = 0.05 + (0.37 * 0.20)   = 0.124
        illness_delay      = 1 + int(0.37 * 4)       = 2
        randomize_sources  = 0.37 < 0.4              = False
    """
    dec = max(0.0, min(1.0, float(deception_level)))
    caps = DECEPTION_CAPS.get(task_id, DECEPTION_CAPS[3])
    return {
        "false_signal_count": int(dec * caps["max_false_signals"]),
        "sensor_noise_std": round(
            base_config.sensor_noise_std
            + dec * (float(caps["max_noise"]) - base_config.sensor_noise_std),
            4,
        ),
        "illness_delay": base_config.illness_delay + int(
            dec * (int(caps["max_delay"]) - base_config.illness_delay)
        ),
        "randomize_sources": dec > 0.4,
    }


def build_task_registry() -> dict[int, TaskConfig]:

    return {
        1: TaskConfig(
            task_id=1,
            name="easy",
            description="Single contaminated farm, clean sensors, and enough budget to inspect before acting.",
            goal="Find the contaminated source farm and stop unsafe batches before they reach retailers.",
            node_specs=(
                NodeSpec("farm_a", "farm", ("processing_p1",)),
                NodeSpec("farm_b", "farm", ("processing_p1",)),
                NodeSpec("processing_p1", "processing", ("warehouse_w1",)),
                NodeSpec("warehouse_w1", "warehouse", ("retailer_r1", "retailer_r2")),
                NodeSpec("retailer_r1", "retailer", ()),
                NodeSpec("retailer_r2", "retailer", ()),
            ),
            n_sources=1,
            sensor_noise_std=0.05,
            illness_delay=1,
            lab_budget=10,
            recall_budget=100,
            max_steps=48,
            false_signal_count=0,
            reseed_interval=None,
            batches_per_farm=3,
            edge_weight_low=0.18,
            edge_weight_high=0.28,
            contamination_threshold=0.35,
            enable_trace=True,
            enable_hold=False,
            score_weights=ScoreWeights(0.50, 0.30, 0.10, 0.10),
        ),
        2: TaskConfig(
            task_id=2,
            name="medium",
            description="Two contamination sources, delayed retailer signals, and limited lab budget.",
            goal="Trace multiple contaminated branches and prioritize containment with only a few tests.",
            node_specs=(
                NodeSpec("farm_a", "farm", ("processing_p1",)),
                NodeSpec("farm_b", "farm", ("processing_p1", "processing_p2")),
                NodeSpec("farm_c", "farm", ("processing_p2",)),
                NodeSpec("processing_p1", "processing", ("warehouse_w1", "warehouse_w2")),
                NodeSpec("processing_p2", "processing", ("warehouse_w2",)),
                NodeSpec("warehouse_w1", "warehouse", ("retailer_r1", "retailer_r2")),
                NodeSpec("warehouse_w2", "warehouse", ("retailer_r3", "retailer_r4", "retailer_r5")),
                NodeSpec("retailer_r1", "retailer", ()),
                NodeSpec("retailer_r2", "retailer", ()),
                NodeSpec("retailer_r3", "retailer", ()),
                NodeSpec("retailer_r4", "retailer", ()),
                NodeSpec("retailer_r5", "retailer", ()),
            ),
            n_sources=2,
            sensor_noise_std=0.15,
            illness_delay=3,
            lab_budget=6,
            recall_budget=60,
            max_steps=60,
            false_signal_count=0,
            reseed_interval=None,
            batches_per_farm=3,
            edge_weight_low=0.16,
            edge_weight_high=0.30,
            contamination_threshold=0.33,
            enable_trace=True,
            enable_hold=False,
            score_weights=ScoreWeights(0.40, 0.25, 0.15, 0.20),
        ),
        3: TaskConfig(
            task_id=3,
            name="hard",
            description="Adversarial false signals, noisy sensors, delayed reports, and re-seeding contamination.",
            goal="Contain a deceptive multi-source outbreak without collapsing public trust or overspending.",
            node_specs=(
                NodeSpec("farm_a", "farm", ("processing_p1", "processing_p2")),
                NodeSpec("farm_b", "farm", ("processing_p1",)),
                NodeSpec("farm_c", "farm", ("processing_p2", "processing_p3")),
                NodeSpec("farm_d", "farm", ("processing_p3",)),
                NodeSpec("farm_e", "farm", ("processing_p4",)),
                NodeSpec("processing_p1", "processing", ("warehouse_w1", "warehouse_w2")),
                NodeSpec("processing_p2", "processing", ("warehouse_w2",)),
                NodeSpec("processing_p3", "processing", ("warehouse_w3", "warehouse_w4")),
                NodeSpec("processing_p4", "processing", ("warehouse_w4",)),
                NodeSpec("warehouse_w1", "warehouse", ("retailer_r1", "retailer_r2")),
                NodeSpec("warehouse_w2", "warehouse", ("retailer_r3", "retailer_r4")),
                NodeSpec("warehouse_w3", "warehouse", ("retailer_r5", "retailer_r6")),
                NodeSpec("warehouse_w4", "warehouse", ("retailer_r7",)),
                NodeSpec("retailer_r1", "retailer", ()),
                NodeSpec("retailer_r2", "retailer", ()),
                NodeSpec("retailer_r3", "retailer", ()),
                NodeSpec("retailer_r4", "retailer", ()),
                NodeSpec("retailer_r5", "retailer", ()),
                NodeSpec("retailer_r6", "retailer", ()),
                NodeSpec("retailer_r7", "retailer", ()),
            ),
            n_sources=2,
            sensor_noise_std=0.25,
            illness_delay=5,
            lab_budget=4,
            recall_budget=40,
            max_steps=72,
            false_signal_count=3,
            reseed_interval=10,
            batches_per_farm=4,
            edge_weight_low=0.14,
            edge_weight_high=0.32,
            contamination_threshold=0.30,
            enable_trace=True,
            enable_hold=False,
            score_weights=ScoreWeights(0.35, 0.20, 0.15, 0.30),
        ),
    }


TASKS = build_task_registry()


def get_task_config(task_id: int = 1) -> TaskConfig:
    try:
        return TASKS[int(task_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported task_id: {task_id}") from exc
