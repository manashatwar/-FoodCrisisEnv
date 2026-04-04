from __future__ import annotations

import os
import random
from typing import Generic, TypeVar

if os.getenv("FOODCRISIS_STANDALONE") == "1":
    ActionT = TypeVar("ActionT")
    ObservationT = TypeVar("ObservationT")
    StateT = TypeVar("StateT")

    class Environment(Generic[ActionT, ObservationT, StateT]):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self) -> None:
            pass

        def _apply_transform(self, observation: ObservationT) -> ObservationT:
            return observation

        def _reset_rubric(self) -> None:
            return None
else:
    try:
        from openenv.core.env_server import Environment
    except ImportError:  # pragma: no cover
        from openenv_core.env_server import Environment

from irce.models import (
    BatchRecord,
    FoodCrisisAction,
    FoodCrisisObservation,
    FoodCrisisState,
    IllnessReport,
    NodeState,
    PendingInspection,
    SUPPORTED_ACTIONS,
)
from irce.rewards import compute_step_reward
from irce.tasks import TaskConfig, get_task_config


class FoodCrisisEnv(Environment[FoodCrisisAction, FoodCrisisObservation, FoodCrisisState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: int = 1, seed: int | None = 7) -> None:
        super().__init__()
        self.task_id = task_id
        self.task_config = get_task_config(task_id)
        self._default_seed = 7 if seed is None else seed
        self._rng = random.Random(self._default_seed)
        self._episode_index = 0
        self.episode_log: list[dict[str, object]] = []
        self._downstream_map: dict[str, tuple[str, ...]] = {}
        self._upstream_map: dict[str, list[str]] = {}
        self._node_types: dict[str, str] = {}
        self._state = FoodCrisisState(episode_id=self._build_episode_id(self._default_seed))
        self._set_task(self.task_id)

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: object) -> FoodCrisisObservation:
        requested_task_id = kwargs.pop("task_id", self.task_id)
        self.task_id = int(requested_task_id)
        self._set_task(self.task_id)
        self._reset_rubric()

        episode_seed = self._default_seed if seed is None else seed
        self._rng = random.Random(episode_seed)
        self._episode_index += 1
        self.episode_log = []

        state = FoodCrisisState(
            episode_id=episode_id or self._build_episode_id(episode_seed),
            goal=self.task_config.goal,
            task_name=self.task_config.name,
            timestep=0,
            lab_budget=self.task_config.lab_budget,
            recall_budget=self.task_config.recall_budget,
            public_trust=1.0,
            compat_error_type="TRANSIENT",
            compat_same_error_count=0,
        )

        for spec in self.task_config.node_specs:
            state.nodes[spec.node_id] = NodeState(
                node_id=spec.node_id,
                node_type=spec.node_type,
                sensor_reading=0.0,
                quarantined=False,
                batch_ids=[],
                connected_to=list(spec.downstream),
            )
            state.true_contamination[spec.node_id] = 0.0
            state.quarantine_status[spec.node_id] = False
            state.alert_timers[spec.node_id] = 0
            for downstream in spec.downstream:
                state.edge_weights[self._edge_key(spec.node_id, downstream)] = round(
                    self._rng.uniform(self.task_config.edge_weight_low, self.task_config.edge_weight_high),
                    3,
                )

        state.source_nodes = sorted(self._rng.sample(list(self.task_config.farm_ids), self.task_config.n_sources))
        for source in state.source_nodes:
            state.true_contamination[source] = 1.0

        self._seed_batches(state)
        self._assign_false_signals(state)
        self._refresh_nodes(state)
        self._state = state
        return self._build_observation(
            tool_result="AMBIGUOUS",
            last_action_error=False,
            reward=0.0,
            done=False,
            note="Initial outbreak state prepared.",
        )

    def step(self, action: FoodCrisisAction, timeout_s: float | None = None, **kwargs: object) -> FoodCrisisObservation:
        del timeout_s, kwargs

        if not self._state.nodes or not self._state.true_contamination:
            self.reset(seed=self._default_seed, task_id=self.task_id)

        state = self._state
        config = self.task_config
        state.timestep += 1

        requested_action = action.action_type
        verb, target, resolved_action = self._resolve_action(action)
        state.last_action = resolved_action

        correct_quarantine = False
        source_quarantine = False
        contaminated_quarantine = False
        wrong_quarantine = False
        correct_recall = False
        useful_inspection = False
        wasted_action = False
        wrong_recall = False
        helpful_action = False
        trace_performed = False
        last_action_error = False
        note = ""

        if verb not in SUPPORTED_ACTIONS:
            wasted_action = True
            last_action_error = True
            note = f"Unsupported action '{requested_action}'. Valid actions: {', '.join(sorted(SUPPORTED_ACTIONS))}"
        elif verb == "WAIT":
            note = "Agent waited for additional evidence."
        elif verb == "INSPECT":
            if not target or target not in state.nodes:
                wasted_action = True
                last_action_error = True
                note = "Inspect target was missing or unknown."
            elif state.lab_budget <= 0:
                wasted_action = True
                last_action_error = True
                note = "No lab budget remained for inspection."
            elif target in state.lab_results or any(pending.node_id == target for pending in state.pending_inspections):
                wasted_action = True
                last_action_error = True
                note = f"Inspection of {target} was redundant."
            else:
                state.lab_budget -= 1
                state.pending_inspections.append(
                    PendingInspection(
                        node_id=target,
                        due_timestep=state.timestep + 1,
                        contaminated=state.true_contamination[target] >= config.contamination_threshold,
                    )
                )
                useful_inspection = True
                helpful_action = True
                note = f"Inspection of {target} was scheduled."
        elif verb == "QUARANTINE":
            if not target or target not in state.nodes:
                wasted_action = True
                last_action_error = True
                note = "Quarantine target was missing or unknown."
            else:
                state.total_actions += 1
                state.quarantine_status[target] = True
                state.nodes[target].quarantined = True
                if state.true_contamination[target] >= config.contamination_threshold:
                    state.correct_actions += 1
                    correct_quarantine = True
                    if target in state.source_nodes:
                        source_quarantine = True
                        note = f"Quarantine on source node {target} blocked the outbreak origin."
                    else:
                        contaminated_quarantine = True
                        note = f"Quarantine on {target} blocked a contaminated node."
                    helpful_action = True
                else:
                    state.false_positive_count += 1
                    wrong_quarantine = True
                    last_action_error = True
                    note = f"Quarantine on clean node {target} reduced trust."
        elif verb == "LIFT":
            if not target or target not in state.nodes:
                wasted_action = True
                last_action_error = True
                note = "Lift target was missing or unknown."
            else:
                state.total_actions += 1
                state.quarantine_status[target] = False
                state.nodes[target].quarantined = False
                if state.lab_results.get(target) == "clean":
                    state.correct_actions += 1
                    helpful_action = True
                note = f"Quarantine lifted for {target}."
        elif verb == "RECALL":
            if not target or target not in state.batch_records:
                wasted_action = True
                last_action_error = True
                note = "Recall target was missing or unknown."
            elif state.recall_budget < 10:
                wasted_action = True
                last_action_error = True
                note = "Recall budget was exhausted."
            else:
                batch = state.batch_records[target]
                if batch.recalled or batch.delivered:
                    wasted_action = True
                    last_action_error = True
                    note = f"Batch {target} was already inactive."
                else:
                    state.total_actions += 1
                    state.recall_budget -= 10
                    if target in state.nodes[batch.current_node].batch_ids:
                        state.nodes[batch.current_node].batch_ids.remove(target)
                    batch.recalled = True
                    if batch.contaminated:
                        state.correct_actions += 1
                        correct_recall = True
                        helpful_action = True
                        note = f"Contaminated batch {target} was recalled."
                    else:
                        wrong_recall = True
                        last_action_error = True
                        state.false_positive_count += 1
                        state.public_trust = max(0.0, round(state.public_trust - 0.05, 3))
                        note = f"Clean batch {target} was recalled by mistake."
        elif verb == "TRACE":
            if not config.enable_trace:
                wasted_action = True
                last_action_error = True
                note = "TRACE is disabled for this task."
            elif not target or target not in state.batch_records:
                wasted_action = True
                last_action_error = True
                note = "Trace target was missing or unknown."
            else:
                batch = state.batch_records[target]
                state.traced_batches[target] = list(batch.path_taken)
                trace_performed = True
                helpful_action = True
                note = f"Trace for {target}: {' -> '.join(batch.path_taken)}"
        elif verb == "ALERT":
            if not target or target not in state.nodes or state.nodes[target].node_type != "retailer":
                wasted_action = True
                last_action_error = True
                note = "Alerts must target a retailer."
            else:
                state.total_actions += 1
                state.alert_timers[target] = 3
                state.public_trust = max(0.0, round(state.public_trust - 0.08, 3))
                if self._retailer_risk(target) >= 0.5:
                    state.correct_actions += 1
                    helpful_action = True
                note = f"Public alert issued for {target}."

        self._apply_clean_quarantine_trust_drain(state)
        self._propagate_contamination(state)
        self._reseed_if_needed(state)
        new_contaminated_shipments, prevented_shipments = self._move_batches(state)
        new_illness_cases = self._deliver_retail_exposure(state)
        released_lab_results = self._release_due_lab_results(state)
        released_reports = self._release_due_illness_reports(state)
        self._tick_alerts(state)
        self._refresh_nodes(state)

        reward_breakdown = compute_step_reward(
            new_contaminated_shipments=new_contaminated_shipments,
            new_illness_cases=new_illness_cases,
            source_quarantine=source_quarantine,
            contaminated_quarantine=contaminated_quarantine,
            wrong_quarantine=wrong_quarantine,
            correct_recall=correct_recall,
            wrong_recall=wrong_recall,
            prevented_shipments=prevented_shipments,
            useful_inspection=useful_inspection,
            wasted_action=wasted_action,
            active_uncontained_sources=self._active_uncontained_sources(state),
            trace_performed=trace_performed,
        )

        tool_result = self._derive_tool_result(
            last_action_error=last_action_error,
            helpful_action=helpful_action,
            useful_inspection=useful_inspection,
            released_lab_results=released_lab_results,
            released_reports=released_reports,
            note=note,
        )
        done = self._check_done(state)
        observation = self._build_observation(
            tool_result=tool_result,
            last_action_error=last_action_error,
            reward=reward_breakdown.total,
            done=done,
            note=note,
        )

        state.history.append(f"t={state.timestep} {resolved_action} -> {tool_result}")
        state.history = state.history[-3:]
        observation.history_tail = list(state.history)
        observation.status_summary = observation.natural_language_summary
        state.last_reward = reward_breakdown.total
        state.last_outcome = tool_result

        self.episode_log.append(
            {
                "task_id": self.task_id,
                "task_name": config.name,
                "requested_action": requested_action,
                "action": resolved_action,
                "reward": reward_breakdown.total,
                "tool_result": tool_result,
                "timestep": state.timestep,
                "step_count": state.timestep,
                "max_steps": config.max_steps,
                "correct_actions": state.correct_actions,
                "total_actions": state.total_actions,
                "public_trust": round(state.public_trust, 3),
                "total_contaminated_batches": state.total_contaminated_batches,
                "exposed_contaminated_batches": state.exposed_contaminated_batches,
                "contained_at_step": state.contained_at_step,
                "budget_remaining": observation.budget_remaining,
                "progress": observation.progress_hint,
                "lab_budget": state.lab_budget,
                "recall_budget": state.recall_budget,
                "new_contaminated_shipments": new_contaminated_shipments,
                "prevented_shipments": prevented_shipments,
                "new_illness_cases": new_illness_cases,
                "done": done,
            }
        )
        return observation

    @property
    def state(self) -> FoodCrisisState:
        return self._state

    def _set_task(self, task_id: int) -> None:
        self.task_config = get_task_config(task_id)
        self._downstream_map = {spec.node_id: spec.downstream for spec in self.task_config.node_specs}
        self._node_types = {spec.node_id: spec.node_type for spec in self.task_config.node_specs}
        self._upstream_map = {node_id: [] for node_id in self.task_config.node_ids}
        for source, downstream_nodes in self._downstream_map.items():
            for downstream in downstream_nodes:
                self._upstream_map[downstream].append(source)

    def _seed_batches(self, state: FoodCrisisState) -> None:
        for farm_id in self.task_config.farm_ids:
            for _ in range(self.task_config.batches_per_farm):
                state.batch_counter += 1
                batch_id = f"{farm_id}_batch_{state.batch_counter:03d}"
                path = self._sample_path(farm_id)
                contaminated = farm_id in state.source_nodes
                batch = BatchRecord(
                    batch_id=batch_id,
                    origin_node=farm_id,
                    current_node=farm_id,
                    path_taken=[farm_id],
                    planned_path=path,
                    contaminated=contaminated,
                )
                state.batch_records[batch_id] = batch
                state.nodes[farm_id].batch_ids.append(batch_id)
                if contaminated:
                    state.total_contaminated_batches += 1

    def _assign_false_signals(self, state: FoodCrisisState) -> None:
        if self.task_config.false_signal_count <= 0:
            return
        candidates = [node_id for node_id in self.task_config.node_ids if node_id not in state.source_nodes]
        count = min(self.task_config.false_signal_count, len(candidates))
        state.false_signal_nodes = sorted(self._rng.sample(candidates, count))

    def _sample_path(self, origin: str) -> list[str]:
        path = [origin]
        current = origin
        visited = {origin}
        while self._downstream_map[current]:
            downstream = self._rng.choice(list(self._downstream_map[current]))
            path.append(downstream)
            if downstream in visited:
                break
            visited.add(downstream)
            current = downstream
            if self._node_types[current] == "retailer":
                break
        return path

    def _resolve_action(self, action: FoodCrisisAction) -> tuple[str, str | None, str]:
        verb = action.verb
        target = action.target

        if verb == "WAIT":
            return "WAIT", None, "WAIT"
        if verb in {"INSPECT", "QUARANTINE"} and not target:
            target = self._select_suspect_node()
        elif verb == "LIFT" and not target:
            target = self._select_lift_target()
        elif verb == "RECALL" and not target:
            target = self._select_batch_to_recall()
        elif verb == "TRACE" and not target:
            target = self._select_batch_to_trace()
        elif verb == "ALERT" and not target:
            target = self._select_alert_target()
        return verb, target, f"{verb} {target}".strip()
    def _select_suspect_node(self) -> str | None:
        best_node = None
        best_score = float("-inf")
        report_nodes = {report.retailer_id for report in self._state.illness_reports}

        for node_id, node in self._state.nodes.items():
            sensor = node.sensor_reading
            node_bonus = {"farm": 0.18, "processing": 0.12, "warehouse": 0.08, "retailer": 0.02}[node.node_type]
            report_bonus = 0.0
            if any(retailer in report_nodes for retailer in self._descendant_retailers(node_id)):
                report_bonus = 0.25
            quarantine_penalty = -0.1 if node.quarantined else 0.0
            risk = sensor + node_bonus + report_bonus + quarantine_penalty
            if risk > best_score:
                best_score = risk
                best_node = node_id
        return best_node

    def _select_lift_target(self) -> str | None:
        for node_id in self.task_config.node_ids:
            if self._state.quarantine_status.get(node_id) and self._state.lab_results.get(node_id) == "clean":
                return node_id
        for node_id in self.task_config.node_ids:
            if self._state.quarantine_status.get(node_id):
                return node_id
        return None

    def _select_alert_target(self) -> str | None:
        best_retailer = None
        best_score = 0.0
        for retailer_id in self.task_config.retailer_ids:
            score = self._retailer_risk(retailer_id)
            if score > best_score:
                best_score = score
                best_retailer = retailer_id
        return best_retailer if best_score >= 0.35 else None

    def _select_batch_to_recall(self) -> str | None:
        best_batch_id = None
        best_score = float("-inf")
        for batch_id, batch in self._state.batch_records.items():
            if batch.recalled or batch.delivered:
                continue
            current_node = batch.current_node
            node_type = self._node_types[current_node]
            proximity = {"retailer": 0.35, "warehouse": 0.22, "processing": 0.12, "farm": 0.04}[node_type]
            lab_bonus = 0.3 if self._state.lab_results.get(current_node) == "contaminated" else 0.0
            contamination_hint = self._state.nodes[current_node].sensor_reading
            score = proximity + lab_bonus + contamination_hint
            if score > best_score:
                best_score = score
                best_batch_id = batch_id
        return best_batch_id

    def _select_batch_to_trace(self) -> str | None:
        return self._select_batch_to_recall()

    def _descendant_retailers(self, node_id: str) -> list[str]:
        retailers: list[str] = []
        stack = list(self._downstream_map[node_id])
        seen: set[str] = set()
        while stack:
            candidate = stack.pop()
            if candidate in seen:
                continue
            seen.add(candidate)
            if self._node_types[candidate] == "retailer":
                retailers.append(candidate)
            else:
                stack.extend(self._downstream_map[candidate])
        return retailers

    def _retailer_risk(self, retailer_id: str) -> float:
        sensor = self._state.nodes[retailer_id].sensor_reading
        report_bonus = sum(report.case_count for report in self._state.illness_reports if report.retailer_id == retailer_id) * 0.08
        quarantine_penalty = -0.05 if self._state.quarantine_status.get(retailer_id) else 0.0
        return sensor + report_bonus + quarantine_penalty

    def _apply_clean_quarantine_trust_drain(self, state: FoodCrisisState) -> None:
        clean_quarantines = 0
        for node_id, quarantined in state.quarantine_status.items():
            if quarantined and state.true_contamination[node_id] < self.task_config.contamination_threshold:
                clean_quarantines += 1
        if clean_quarantines:
            state.public_trust = max(0.0, round(state.public_trust - (0.05 * clean_quarantines), 3))

    def _propagate_contamination(self, state: FoodCrisisState) -> None:
        new_levels = dict(state.true_contamination)
        for node_id in self.task_config.node_ids:
            inflow = 0.0
            for upstream in self._upstream_map[node_id]:
                if state.quarantine_status[upstream]:
                    continue
                inflow += state.true_contamination[upstream] * state.edge_weights[self._edge_key(upstream, node_id)]
            new_levels[node_id] = round(min(1.0, state.true_contamination[node_id] + inflow), 3)
        state.true_contamination = new_levels

    def _reseed_if_needed(self, state: FoodCrisisState) -> None:
        if self.task_config.reseed_interval is None:
            return
        if state.timestep == 0 or state.timestep % self.task_config.reseed_interval != 0:
            return

        for source in state.source_nodes:
            state.true_contamination[source] = max(state.true_contamination[source], 1.0)
            state.batch_counter += 1
            batch_id = f"{source}_batch_{state.batch_counter:03d}"
            batch = BatchRecord(
                batch_id=batch_id,
                origin_node=source,
                current_node=source,
                path_taken=[source],
                planned_path=self._sample_path(source),
                contaminated=True,
            )
            state.batch_records[batch_id] = batch
            state.nodes[source].batch_ids.append(batch_id)
            state.total_contaminated_batches += 1

    def _move_batches(self, state: FoodCrisisState) -> tuple[int, int]:
        contaminated_shipments = 0
        prevented_shipments = 0
        for batch in state.batch_records.values():
            if batch.recalled or batch.delivered:
                continue
            current_node = batch.current_node
            current_index = batch.planned_path.index(current_node)
            if current_index >= len(batch.planned_path) - 1:
                continue
            if state.quarantine_status[current_node]:
                if batch.contaminated:
                    prevented_shipments += 1
                continue

            next_node = batch.planned_path[current_index + 1]
            if batch.batch_id in state.nodes[current_node].batch_ids:
                state.nodes[current_node].batch_ids.remove(batch.batch_id)
            state.nodes[next_node].batch_ids.append(batch.batch_id)
            batch.current_node = next_node
            batch.path_taken.append(next_node)
            if batch.contaminated:
                contaminated_shipments += 1
        state.contaminated_shipments += contaminated_shipments
        return contaminated_shipments, prevented_shipments

    def _deliver_retail_exposure(self, state: FoodCrisisState) -> int:
        new_cases = 0
        for batch in state.batch_records.values():
            if batch.recalled or batch.delivered:
                continue
            current_node = batch.current_node
            if self._node_types[current_node] != "retailer":
                continue
            if state.quarantine_status[current_node] or state.alert_timers[current_node] > 0:
                continue
            if not batch.contaminated:
                continue

            if batch.batch_id in state.nodes[current_node].batch_ids:
                state.nodes[current_node].batch_ids.remove(batch.batch_id)
            batch.delivered = True
            state.exposed_contaminated_batches += 1
            case_count = 1 + int(state.true_contamination[current_node] >= 0.8)
            new_cases += case_count
            state.cumulative_illness_cases += case_count
            state.pending_illness_reports.append(
                IllnessReport(
                    retailer_id=current_node,
                    case_count=case_count,
                    timestep_reported=state.timestep + self.task_config.illness_delay,
                    report_text=f"{case_count} illness cases reported at {current_node} (reported hour {state.timestep + self.task_config.illness_delay})",
                )
            )
        return new_cases

    def _release_due_lab_results(self, state: FoodCrisisState) -> dict[str, str]:
        released: dict[str, str] = {}
        remaining: list[PendingInspection] = []
        for pending in state.pending_inspections:
            if pending.due_timestep <= state.timestep:
                released[pending.node_id] = "contaminated" if pending.contaminated else "clean"
            else:
                remaining.append(pending)
        state.pending_inspections = remaining
        state.lab_results.update(released)
        return released

    def _release_due_illness_reports(self, state: FoodCrisisState) -> list[IllnessReport]:
        released: list[IllnessReport] = []
        remaining: list[IllnessReport] = []
        for report in state.pending_illness_reports:
            if report.timestep_reported <= state.timestep:
                released.append(report)
            else:
                remaining.append(report)
        state.pending_illness_reports = remaining
        state.illness_reports = (state.illness_reports + released)[-6:]
        return released

    def _tick_alerts(self, state: FoodCrisisState) -> None:
        for node_id, timer in list(state.alert_timers.items()):
            if timer > 0:
                state.alert_timers[node_id] = timer - 1

    def _refresh_nodes(self, state: FoodCrisisState) -> None:
        sensor_readings = self._generate_sensor_readings(state)
        for node_id, node in state.nodes.items():
            node.sensor_reading = sensor_readings[node_id]
            node.quarantined = state.quarantine_status[node_id]
            node.batch_ids = [
                batch_id
                for batch_id in node.batch_ids
                if batch_id in state.batch_records
                and not state.batch_records[batch_id].recalled
                and not state.batch_records[batch_id].delivered
            ]

    def _generate_sensor_readings(self, state: FoodCrisisState) -> dict[str, float]:
        sensor_readings: dict[str, float] = {}
        for node_id in self.task_config.node_ids:
            noise = self._rng.gauss(0.0, self.task_config.sensor_noise_std)
            reading = state.true_contamination[node_id] + noise
            if node_id in state.false_signal_nodes and state.true_contamination[node_id] < 0.2:
                reading += 0.45
            sensor_readings[node_id] = round(min(1.0, max(0.0, reading)), 3)
        return sensor_readings
    def _contamination_spreading(self, state: FoodCrisisState) -> bool:
        return any(
            batch.contaminated and not batch.recalled and not batch.delivered
            for batch in state.batch_records.values()
        )

    def _active_uncontained_sources(self, state: FoodCrisisState) -> int:
        return sum(
            1
            for source in state.source_nodes
            if not state.quarantine_status.get(source, False)
            and state.true_contamination.get(source, 0.0) >= self.task_config.contamination_threshold
        )

    def _check_done(self, state: FoodCrisisState) -> bool:
        contained = self._is_contained(state)
        if contained and state.contained_at_step is None:
            state.contained_at_step = state.timestep
        return contained or state.timestep >= self.task_config.max_steps or state.recall_budget <= 0

    def _is_contained(self, state: FoodCrisisState) -> bool:
        all_sources_quarantined = all(state.quarantine_status.get(source, False) for source in state.source_nodes)
        active_contaminated_batches = any(
            batch.contaminated and not batch.recalled and not batch.delivered
            for batch in state.batch_records.values()
        )
        return all_sources_quarantined and not active_contaminated_batches

    def _derive_tool_result(
        self,
        *,
        last_action_error: bool,
        helpful_action: bool,
        useful_inspection: bool,
        released_lab_results: dict[str, str],
        released_reports: list[IllnessReport],
        note: str,
    ) -> str:
        if last_action_error:
            return "ERROR"
        if helpful_action or useful_inspection or released_lab_results or released_reports:
            return "SUCCESS"
        if note:
            return "AMBIGUOUS"
        return "AMBIGUOUS"

    def _build_observation(
        self,
        *,
        tool_result: str,
        last_action_error: bool,
        reward: float,
        done: bool,
        note: str,
    ) -> FoodCrisisObservation:
        state = self._state
        sensor_readings = {node_id: node.sensor_reading for node_id, node in state.nodes.items()}
        compat_error_type = self._derive_compat_error_type(state, sensor_readings)
        if compat_error_type == state.compat_error_type:
            state.compat_same_error_count += 1
        else:
            state.compat_same_error_count = 0
            state.compat_error_type = compat_error_type

        observation = FoodCrisisObservation(
            timestep=state.timestep,
            nodes=list(state.nodes.values()),
            sensor_readings=sensor_readings,
            illness_reports=list(state.illness_reports),
            quarantine_status=dict(state.quarantine_status),
            lab_results=dict(state.lab_results),
            traced_batches={batch_id: list(path) for batch_id, path in state.traced_batches.items()},
            lab_budget=state.lab_budget,
            recall_budget=state.recall_budget,
            public_trust=round(state.public_trust, 3),
            natural_language_summary=self._build_nl_summary(state, note),
            goal=self.task_config.goal,
            tool_result=tool_result,
            error_type=compat_error_type,
            same_error_count=state.compat_same_error_count,
            budget_remaining=self._normalized_budget_remaining(state),
            step_count=state.timestep,
            last_action_error=last_action_error,
            active_tool="primary",
            cooldown_remaining=self._pending_lab_cooldown(state),
            progress_hint=self._progress_hint(state),
            history_tail=list(state.history),
            status_summary="",
            reward=round(reward, 3),
            done=done,
        )
        observation.status_summary = observation.natural_language_summary
        return self._apply_transform(observation)

    def _derive_compat_error_type(self, state: FoodCrisisState, sensor_readings: dict[str, float]) -> str:
        max_sensor = max(sensor_readings.values()) if sensor_readings else 0.0
        if state.lab_budget <= 0 or state.recall_budget < 10:
            return "RATE_LIMIT"
        if state.illness_reports or max_sensor >= 0.6:
            return "HARD"
        return "TRANSIENT"

    def _normalized_budget_remaining(self, state: FoodCrisisState) -> float:
        lab_ratio = state.lab_budget / max(1, self.task_config.lab_budget)
        recall_ratio = state.recall_budget / max(1, self.task_config.recall_budget)
        return round(max(0.0, min(1.0, (lab_ratio + recall_ratio + state.public_trust) / 3.0)), 3)

    def _pending_lab_cooldown(self, state: FoodCrisisState) -> int:
        if not state.pending_inspections:
            return 0
        soonest_due = min(pending.due_timestep for pending in state.pending_inspections)
        return max(0, soonest_due - state.timestep)

    def _progress_hint(self, state: FoodCrisisState) -> float:
        if self._is_contained(state):
            return 1.0
        source_quarantine_ratio = sum(1 for source in state.source_nodes if state.quarantine_status.get(source, False)) / max(1, len(state.source_nodes))
        active_contaminated = sum(
            1
            for batch in state.batch_records.values()
            if batch.contaminated and not batch.recalled and not batch.delivered
        )
        batch_control_ratio = 1.0 - (active_contaminated / max(1, state.total_contaminated_batches))
        return round(max(0.0, min(1.0, (0.55 * source_quarantine_ratio) + (0.45 * batch_control_ratio))), 3)

    def _build_nl_summary(self, state: FoodCrisisState, note: str) -> str:
        sensor_pairs = sorted(state.nodes.items(), key=lambda item: item[1].sensor_reading, reverse=True)
        risky_nodes = ", ".join(f"{node_id}={node.sensor_reading:.2f}" for node_id, node in sensor_pairs[:3]) or "none"
        source_candidates = [
            (node_id, node)
            for node_id, node in sensor_pairs
            if node.node_type in {"farm", "processing"}
        ]
        suspected_sources = ", ".join(
            f"{node_id}={node.sensor_reading:.2f}"
            for node_id, node in source_candidates[:2]
        ) or "none"
        quarantines = [node_id for node_id, quarantined in state.quarantine_status.items() if quarantined]
        quarantine_text = ", ".join(quarantines[:4]) if quarantines else "none"
        reports = "; ".join(report.report_text for report in state.illness_reports[-2:]) if state.illness_reports else "none"
        lab_updates = ", ".join(f"{node_id}:{result}" for node_id, result in list(state.lab_results.items())[-3:]) or "none"
        pending_labs = ", ".join(f"{pending.node_id}@{pending.due_timestep}" for pending in state.pending_inspections[:3]) or "none"
        trace_updates = "; ".join(
            f"{batch_id}:{'->'.join(path)}"
            for batch_id, path in list(state.traced_batches.items())[-2:]
        ) or "none"
        contaminated_nodes = ", ".join(
            sorted(node_id for node_id, result in state.lab_results.items() if result == "contaminated")[:3]
        ) or "none"
        report_retailers = {report.retailer_id for report in state.illness_reports}
        hint_parts: list[str] = []
        if source_candidates and source_candidates[0][1].sensor_reading >= 0.6:
            hint_parts.append("Hint: recent upstream spikes may indicate source contamination.")
        if len(report_retailers) >= 2:
            hint_parts.append("Hint: multiple downstream signals may suggest a shared source.")
        hint_text = " ".join(hint_parts)
        false_signal_hint = "Warning: some sensor spikes may be false positives in this task." if state.false_signal_nodes else ""
        return (
            f"Hour {state.timestep} | "
            f"Suspected sources: {suspected_sources}. "
            f"Most risky nodes: {risky_nodes}. "
            f"Recent illness spikes: {reports}. "
            f"Confirmed contamination: {contaminated_nodes}. "
            f"Quarantined nodes: {quarantine_text}. "
            f"Pending lab tests: {pending_labs}. "
            f"Recent lab results: {lab_updates}. "
            f"Recent traces: {trace_updates}. "
            f"Budget status: lab={state.lab_budget}, recall={state.recall_budget}. "
            f"Trust level: {state.public_trust:.2f}. "
            f"{note} {hint_text} {false_signal_hint}"
        ).strip()


    def _edge_key(self, source: str, target: str) -> str:
        return f"{source}->{target}"

    def _build_episode_id(self, seed: int) -> str:
        return f"food-crisis-{seed}-{self._episode_index}"
