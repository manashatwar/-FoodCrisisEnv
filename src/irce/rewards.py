from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    shipment_penalty: float
    illness_penalty: float
    source_quarantine_bonus: float
    contaminated_quarantine_bonus: float
    wrong_quarantine_penalty: float
    correct_recall_bonus: float
    wrong_recall_penalty: float
    prevented_shipment_bonus: float
    useful_inspection_bonus: float
    wasted_action_penalty: float
    active_source_penalty: float
    trace_cost: float
    total: float


def compute_step_reward(
    *,
    new_contaminated_shipments: int,
    new_illness_cases: int,
    source_quarantine: bool,
    contaminated_quarantine: bool,
    wrong_quarantine: bool,
    correct_recall: bool,
    wrong_recall: bool,
    prevented_shipments: int,
    useful_inspection: bool,
    wasted_action: bool,
    active_uncontained_sources: int,
    trace_performed: bool,
) -> RewardBreakdown:
    shipment_penalty = -1.5 * float(new_contaminated_shipments)
    illness_penalty = -0.5 * float(new_illness_cases)
    source_quarantine_bonus = 4.0 if source_quarantine else 0.0
    contaminated_quarantine_bonus = 2.0 if contaminated_quarantine else 0.0
    wrong_quarantine_penalty = -2.0 if wrong_quarantine else 0.0
    correct_recall_bonus = 1.5 if correct_recall else 0.0
    wrong_recall_penalty = -1.0 if wrong_recall else 0.0
    prevented_shipment_bonus = 0.5 * float(prevented_shipments)
    useful_inspection_bonus = 0.3 if useful_inspection else 0.0
    wasted_action_penalty = -0.1 if wasted_action else 0.0
    active_source_penalty = -0.15 * float(active_uncontained_sources)
    trace_cost = -0.1 if trace_performed else 0.0

    total = (
        shipment_penalty
        + illness_penalty
        + source_quarantine_bonus
        + contaminated_quarantine_bonus
        + wrong_quarantine_penalty
        + correct_recall_bonus
        + wrong_recall_penalty
        + prevented_shipment_bonus
        + useful_inspection_bonus
        + wasted_action_penalty
        + active_source_penalty
        + trace_cost
    )

    return RewardBreakdown(
        shipment_penalty=shipment_penalty,
        illness_penalty=illness_penalty,
        source_quarantine_bonus=source_quarantine_bonus,
        contaminated_quarantine_bonus=contaminated_quarantine_bonus,
        wrong_quarantine_penalty=wrong_quarantine_penalty,
        correct_recall_bonus=correct_recall_bonus,
        wrong_recall_penalty=wrong_recall_penalty,
        prevented_shipment_bonus=prevented_shipment_bonus,
        useful_inspection_bonus=useful_inspection_bonus,
        wasted_action_penalty=wasted_action_penalty,
        active_source_penalty=active_source_penalty,
        trace_cost=trace_cost,
        total=round(total, 3),
    )
