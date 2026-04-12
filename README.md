---
title: FoodCrisisEnv
emoji: 🥬
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Food crisis LLM agent benchmark
tags:
  - reinforcement-learning
  - environment
  - benchmark
  - food-safety
  - llm-agent
  - openenv
---

# FoodCrisisEnv

> An OpenEnv benchmark where an agent acts as a food safety investigator tracing contaminated food through a supply chain before it reaches consumers.

FoodCrisisEnv turns outbreak response into a sequential decision problem. The agent sees noisy sensor data, delayed illness reports, limited lab tests, limited recall budget, and a public-trust penalty for overreacting. It must decide when to trace, inspect, quarantine, alert, recall, or wait.

This is designed as a real operations benchmark, not a toy control task. The setting is grounded in FDA food traceability workflows and the FSMA 204 traceability rule, where investigators need to connect contaminated lots, shipment history, and late-arriving health signals under time pressure.

---

## 🌐 Live Demo

Visit the Space URL to open the **Incident Command Dashboard** — a fully interactive web UI where you can:

- 🗺️ Visualize the live supply chain graph with real-time sensor readings and contamination status.
- 🎮 Execute actions manually (INSPECT, QUARANTINE, LIFT, RECALL, TRACE, WAIT).
- 📊 Monitor lab budgets, recall budgets, public trust, and illness reports.
- 🔄 Select Task 1 / 2 / 3 difficulty and reset episodes instantly.



## 🔌 Connect Your Agent

```python
from irce.client import FoodCrisisEnvClient
from irce.models import FoodCrisisAction

BASE_URL = "https://<your-username>-foodcrisisenv.hf.space"

with FoodCrisisEnvClient(BASE_URL).sync() as env:
    obs = env.reset(task_id=1, seed=7)
    while not obs.done:
        action = FoodCrisisAction(action_type="WAIT")  # replace with your policy
        obs = env.step(action)
```

Or install the client directly from this Space:

```bash
pip install git+https://huggingface.co/spaces/<username>/foodcrisisenv
```



## 📡 API Endpoints

| Method | Path      | Description                                                             |
| ------ | --------- | ----------------------------------------------------------------------- |
| `GET`  | `/`       | Interactive web dashboard                                               |
| `GET`  | `/health` | Liveness check                                                          |
| `POST` | `/reset`  | Start new episode — body: `{"task_id": 1, "seed": 7}`                   |
| `POST` | `/step`   | Take one action — body: `{"action": {"action_type": "INSPECT farm_a"}}` |
| `GET`  | `/state`  | Full environment state (observation + metadata)                         |
| `GET`  | `/docs`   | Interactive Swagger UI                                                  |
| `GET`  | `/redoc`  | ReDoc API reference                                                     |

### Example curl calls

```bash
# Health check
curl https://<your-space>.hf.space/health

# Reset to Task 1
curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 7}'

# Take an action
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "TRACE farm_a_batch_001"}}'

# Get current state
curl https://<your-space>.hf.space/state
```

---

## 🎮 Interactive Web UI Guide

The **Incident Command Dashboard** provides a visual interface for manual control and monitoring. Here's what each panel does:

### Left Panel: Mission Control

| Button/Control | What It Does |
|---|---|
| **T1 Easy / T2 Med / T3 Hard** | Select difficulty (Easy: fast reports, clean signal; Medium: more noise; Hard: very noisy, delayed reports) |
| **⟳ Reset Episode** | Start a fresh episode with the selected task difficulty |
| **Progress Ring** | Shows current step / total steps (0-48) and completion % |
| **🔬 Lab Budget** | Remaining lab tests (0-10). Each INSPECT uses 1. Green bar shows available. |
| **📦 Recall Budget** | Remaining recall capacity (0-100). Each RECALL uses allocated budget. Purple bar shows available. |
| **👥 Public Trust** | Current public confidence (0-100%). Unnecessary quarantines reduce this. Green bar. |
| **Total Mission Score** | Cumulative reward across all actions (higher is better) |
| **Latest Reward** | Feedback from your last action (positive = good decision, negative = costly mistake) |

### Center Pane: Supply Chain Graph

- **Node colors**: Green (clean), Amber (suspect: sensor ≥ 0.4), Red (contaminated), Gray (unknown)
- **Icons**: ✓ (clean lab result), ☣ (contaminated), ⚠ (quarantined), ! (high sensor reading ≥ 0.7)
- **Number suffix** (e.g., "3b"): Shows batch count at this node
- **Dashed border**: Node is quarantined
- **Red arrows**: Contamination flow detected
- **Faded arrows**: Quarantined paths (blocked)

Click **↺ Refresh** to force an immediate state update (auto-refreshes every 5s).

### Right Panel: Operations

#### Manual Control Mode (default)

1. **Select Node**: Dropdown lists all visible `farm_a`, `farm_b`, etc.
2. **Action Buttons**:
   - **INSPECT** — Run lab test on selected node (costs 1 lab budget). Result appears in "Lab Results" pills.
   - **TRACE** — Get batch traceback for selected node (free). Shows path from farm to this location.
   - **QUARANTINE** — Block node from receiving/sending food (costs up to 10 recall budget). Dashed border appears on graph.
   - **LIFT** — Remove quarantine from node. Frees recall budget.
   - **RECALL** — Order all batches from node back to farms (costs significant recall budget).
   - **WAIT** — Do nothing this step. Contamination spreads, reports arrive, sensor readings update.

3. **Example Workflow**:
   ```
   Step 1: WAIT (watch for illness reports at retailers)
   Step 2: WAIT (retailer_r1 shows illness)
   Step 3: Select retailer_r1 → TRACE (learn which farms supply it)
   Step 4: Select batch_001 → INSPECT (confirm contamination)
   Step 5: Select farm_a → QUARANTINE (stop spread)
   Step 6: RECALL farm_a (pull all batches from market)
   ```

#### LLM Agent Mode

- Click **🤖 LLM Agent** tab to switch
- Select mode: "Single Step" or "Auto-Play"
- **Single Step**: Agent makes one decision per step (you click each time)
- **Auto-Play**: Agent runs continuously until episode ends
- Agent uses Groq LLM (configured in Space Secrets) to decide actions based on current state

### Panel Readouts

| Panel | Shows |
|---|---|
| **Lab Results Pills** | All test outcomes so far (green = clean, red = contaminated) |
| **Quarantine Pills** | All currently quarantined nodes (yellow 🚫) |
| **Illness Reports** | Which retailers have reports, count of cases |
| **Traced Paths** | Batch lineage showing: `farm → processing → warehouse → retailer` |
| **Episode Log** | Last 30 actions with step, action type, and immediate reward |

---

## Why this benchmark exists

Food recalls are not just detection problems. They are decision problems under uncertainty.

- Sensors are noisy, so the highest reading is not always the real source.
- Illness reports arrive late, so by the time the signal is obvious the contamination may already be downstream.
- Budgets are limited, so inspecting every node or recalling every batch is not realistic.
- Public trust matters, so false quarantines and unnecessary public alerts have real cost.

FoodCrisisEnv models that exact trade-off. The hard task is not "spot the largest number." It is "contain the outbreak quickly without shutting down the wrong branch of the supply chain."

---

## Why RL is needed

FoodCrisisEnv is not just a classification task or a static planning problem.

- **Partial observability**: the true contamination level is hidden, and the agent only sees noisy sensors.
- **Delayed signals**: illness reports arrive after contamination has already moved.
- **Budget trade-offs**: every lab test, recall, and alert competes against limited resources and public trust.
- **Long-horizon consequences**: a bad quarantine decision may look safe now but create trust damage or missed exposure later.

Simple rules work on easy outbreaks but break on the hard task. A greedy "highest sensor first" policy is vulnerable to false spikes, delayed reports, and re-seeding sources. That makes the benchmark a real sequential decision problem rather than a one-step heuristic.

---

## Real-world grounding

The environment is inspired by FDA traceability requirements under FSMA 204 and the Food Traceability Rule. In practical terms, the environment maps to the same operational objects that food safety teams already track:

- `batch_id` plays the role of a traceable lot or shipment identifier.
- Graph edges represent critical tracking events between farms, processors, warehouses, and retailers.
- Node state stores the kind of key data an investigator would use to trace contamination.

**Relevant FDA reference:** [FSMA Rules & Guidance](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-rules-guidance-industry)

---

## Environment Overview

The world is a directed supply-chain graph:

```
farm → processing → warehouse → retailer
```

Contamination starts at one or more source farms and propagates downstream every step. The agent never sees true contamination levels directly — only noisy sensor readings, delayed retailer illness reports, and results from lab inspections it explicitly requests.

Each node stores:

- `node_id` and `node_type`
- Hidden contamination level `true_level` in `[0.0, 1.0]`
- Visible noisy `sensor_reading`
- `quarantined` flag
- `batch_ids` currently at the node
- Downstream links

---

## What the Agent Sees

| Field                      | Meaning                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `timestep`                 | Current hour / step in the outbreak                                                   |
| `nodes`                    | Structured node list with sensor reading, quarantine state, batches, downstream links |
| `sensor_readings`          | Noisy contamination proxy by node                                                     |
| `illness_reports`          | Delayed retailer reports released so far                                              |
| `quarantine_status`        | Current quarantine state per node                                                     |
| `lab_results`              | Exact contamination verdicts for completed inspections                                |
| `traced_batches`           | Supply chain paths for batches that have been traced                                  |
| `lab_budget`               | Remaining lab tests                                                                   |
| `recall_budget`            | Remaining recall budget                                                               |
| `public_trust`             | Starts near 1.0 and drops when the agent overreacts                                   |
| `natural_language_summary` | Judge-friendly and LLM-friendly summary of the current situation                      |

---

## Action Space

| Action                 | Effect                                                                          |
| ---------------------- | ------------------------------------------------------------------------------- |
| `INSPECT <node_id>`    | Spend one lab token to get an exact contamination result on the next step       |
| `QUARANTINE <node_id>` | Block all outbound spread from that node                                        |
| `LIFT <node_id>`       | Remove quarantine and restore flow                                              |
| `RECALL <batch_id>`    | Remove a batch from the supply chain at recall-budget cost                      |
| `ALERT <node_id>`      | Issue a retailer warning that slows exposure but permanently reduces trust      |
| `TRACE <batch_id>`     | Trace a batch backward through the supply chain to identify its source and path |
| `WAIT`                 | Take no direct action and let the system evolve one step                        |

### About TRACE

`TRACE` is a low-cost information action modelling real food-traceability workflows. It returns the complete path a batch took through the supply chain and identifies the origin node.

- **Cost**: -0.1 reward
- **Benefit**: eliminates uncertainty about batch origin and propagation path.
- **Use case**: disambiguate between false signals and real source contamination.
- **Real-world mapping**: corresponds to supplier records and batch traceability data under FSMA 204.

---

## Reward Signal

| Event                                           | Reward    |
| ----------------------------------------------- | --------- |
| Source farm quarantined                         | **+4.0**  |
| Non-source contaminated node quarantined        | **+2.0**  |
| Clean node quarantined (wrong)                  | **−2.0**  |
| Correct contaminated recall                     | **+1.5**  |
| Clean batch recalled (wrong)                    | **−1.0**  |
| Prevented contaminated shipment (per shipment)  | **+0.5**  |
| TRACE action                                    | **−0.1**  |
| Urgency penalty (per active uncontained source) | **−0.15** |

The reward distinguishes finding the **outbreak origin** from chasing downstream symptoms. Quarantining a source farm (+4.0) yields twice the reward of quarantining a downstream processor (+2.0).

---

## Grading

Each episode receives a deterministic final score in `[0.0, 1.0]` from four components:

| Component      | What it measures                                                 |
| -------------- | ---------------------------------------------------------------- |
| `containment`  | How much contaminated product was prevented from reaching retail |
| `precision`    | How often the agent acted correctly instead of overreacting      |
| `speed`        | How early the outbreak was contained                             |
| `public_trust` | How much trust remained at episode end                           |

Task-specific weights shift emphasis from pure containment (easy) toward trust-preserving, budget-aware control (hard).

---

## Task Suite

| Task | Difficulty | Noise | Illness Delay | Lab Budget | Recall Budget | Max Steps |
| :--: | :--------: | ----: | :-----------: | :--------: | :-----------: | :-------: |
|  1   |    Easy    |  0.05 |       1       |     10     |      100      |    48     |
|  2   |   Medium   |  0.15 |       3       |     6      |      60       |    60     |
|  3   |    Hard    |  0.25 |       5       |     4      |      40       |    72     |

**Task 1** — Single source, low noise, fast reports, generous budgets. Solvable with TRACE → INSPECT → QUARANTINE.

**Task 2** — Multi-source outbreak with delayed reports and tighter budgets. Requires prioritization across branches.

**Task 3** — Adversarial false spikes, re-seeding contamination, delayed reports, high trust pressure. Punishes naive "quarantine the highest sensor" strategies.

---

## LLM Prompt Template

The environment is already LLM-friendly because every step includes `natural_language_summary`.

**System prompt:**

```
You are a food safety incident responder controlling FoodCrisisEnv.
Your job is to contain contamination quickly while preserving public trust and limited budgets.

Priorities:
1. TRACE suspicious or already-exposed batches to understand upstream source and propagation path.
2. Confirm likely source nodes with INSPECT when uncertainty is high.
3. QUARANTINE confirmed contaminated source nodes aggressively — finding the origin is worth far more than downstream reaction.
4. RECALL contaminated or highly exposed batches when they are already downstream in the chain.
5. Use ALERT sparingly. It buys time but permanently reduces trust.
6. LIFT quarantine when a node is confirmed clean to restore supply chain and build trust.
7. WAIT only when it is strategically useful, such as waiting for a pending lab result or trace confirmation.

Respond with exactly one valid action string and nothing else.
```

**User prompt:**

```
Observation summary:
{natural_language_summary}

Structured fields:
- timestep: {timestep}
- lab_budget: {lab_budget}
- recall_budget: {recall_budget}
- public_trust: {public_trust}
- lab_results: {lab_results}
- illness_reports: {illness_reports}
- traced_batches: {traced_batches}

Return exactly one action string.
```

---

## Evaluation Interpretation

| Score Range | Meaning                                                                           |
| ----------- | --------------------------------------------------------------------------------- |
| 0.30 – 0.50 | Agent reacts, but containment is inconsistent or too costly                       |
| 0.50 – 0.70 | Agent is operationally useful and beats shallow heuristics                        |
| 0.70 – 0.85 | Strong performance with good containment and disciplined budget use               |
| 0.85+       | Very strong — accurate containment, limited overreaction, good trust preservation |

**Baseline scores** (`seed=7`):

| Policy                    | Task 1 | Task 2 | Task 3 | Average |
| ------------------------- | -----: | -----: | -----: | ------: |
| `FoodCrisisBaselineAgent` |  0.494 |  0.590 |  0.642 |   0.575 |

---

## Common Failure Modes

- **Symptom-chasing without source-finding** — Quarantining downstream nodes (+2.0) instead of tracing back to find the source (+4.0).
- **Over-quarantining without evidence** — Blocking nodes based on sensor spikes alone without TRACE/INSPECT confirmation (−2.0 + trust damage).
- **Ignoring illness signals** — Waiting too long after downstream retailer reports allows batches to keep moving.
- **Wasting lab budget** — Spending inspections on already-clear nodes instead of high-ambiguity decision points.
- **Overusing alerts** — Repeated warnings without traced evidence collapses the trust score.
- **Not recalling contaminated batches** — Once a batch is confirmed downstream, recall (+1.5) is more cost-effective than retroactive containment.

---

## Example Strong Trajectory

```
1. TRACE farm_a_batch_001   → path: farm_a → processing_p1 → warehouse_w1 → retailer_r1
2. INSPECT farm_a           → lab confirms: contaminated
3. QUARANTINE farm_a        → reward: +4.0 (source quarantine)
4. RECALL farm_a_batch_001  → reward: +1.5 (correct recall)
5. WAIT                     → confirm containment
```

---

## Quick Start

```bash
# Install
pip install -e .

# Run inference (uses Groq/OpenAI-compatible API)
python inference.py

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t foodcrisisenv:latest .
docker run --rm -p 7860:7860 --env-file .env foodcrisisenv:latest
```

**Configure model access** (`.env`):

```env
HF_TOKEN=your_groq_or_hf_token_here
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
```

---

## Project Layout

```
.
├── inference.py              ← LLM agent runner
├── Dockerfile                ← Production Docker image
├── src/irce/
│   ├── environment.py        ← Core FoodCrisisEnv simulator
│   ├── grading.py            ← Episode scoring
│   ├── models.py             ← Pydantic types (Action, Observation, State)
│   ├── rewards.py            ← Reward computation
│   ├── tasks.py              ← Task 1/2/3 configurations
│   ├── client.py             ← HTTP client for remote environments
│   └── server/
│       └── app.py            ← FastAPI app with web UI
└── tests/
    └── test_environment.py
```

---

## License

MIT.
