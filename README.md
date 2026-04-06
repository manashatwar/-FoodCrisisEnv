# FoodCrisisEnv

> An OpenEnv benchmark where an agent acts as a food safety investigator tracing contaminated food through a supply chain before it reaches consumers.

FoodCrisisEnv turns outbreak response into a sequential decision problem. The agent sees noisy sensor data, delayed illness reports, limited lab tests, limited recall budget, and a public-trust penalty for overreacting. It must decide when to trace, inspect, quarantine, alert, recall, or wait.

This is designed as a real operations benchmark, not a toy control task. The setting is grounded in FDA food traceability workflows and the FSMA 204 traceability rule, where investigators need to connect contaminated lots, shipment history, and late-arriving health signals under time pressure.

## Why this benchmark exists

Food recalls are not just detection problems. They are decision problems under uncertainty.

- Sensors are noisy, so the highest reading is not always the real source.
- Illness reports arrive late, so by the time the signal is obvious the contamination may already be downstream.
- Budgets are limited, so inspecting every node or recalling every batch is not realistic.
- Public trust matters, so false quarantines and unnecessary public alerts have real cost.

FoodCrisisEnv models that exact trade-off. The hard task is not "spot the largest number." It is "contain the outbreak quickly without shutting down the wrong branch of the supply chain."

## Why this matters

Real outbreak response is messy, time-sensitive, and expensive. Human teams have to connect incomplete supplier data, noisy health signals, and time-delayed reports while deciding whether to shut down a node, recall product, or wait for more evidence. FoodCrisisEnv captures that operational pressure in a compact benchmark, which makes it useful for evaluating whether agents can help with real incident response rather than only solve clean synthetic tasks.

## Why RL is needed

FoodCrisisEnv is not just a classification task or a static planning problem.

- Partial observability: the true contamination level is hidden, and the agent only sees noisy sensors.
- Delayed signals: illness reports arrive after contamination has already moved.
- Budget trade-offs: every lab test, recall, and alert competes against limited resources and public trust.
- Long-horizon consequences: a bad quarantine decision may look safe now but create trust damage or missed exposure later.

Simple rules work on easy outbreaks but break on the hard task. A greedy "highest sensor first" policy is vulnerable to false spikes, delayed reports, and re-seeding sources. That makes the benchmark a real sequential decision problem rather than a one-step heuristic.

## Real-world grounding

The environment is inspired by FDA traceability requirements under FSMA 204 and the Food Traceability Rule. In practical terms, the environment maps to the same operational objects that food safety teams already track:

- `batch_id` plays the role of a traceable lot or shipment identifier.
- graph edges represent critical tracking events between farms, processors, warehouses, and retailers.
- node state stores the kind of key data an investigator would use to trace contamination.

For current regulatory context, FDA's Food Traceability Rule originally targeted January 20, 2026 for compliance and later announced enforcement would not begin before July 20, 2028. FoodCrisisEnv is therefore best understood as a benchmark inspired by the operational problem, not a legal compliance simulator.

Relevant FDA references:

- **FSMA Rules & Guidance:** https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-rules-guidance-industry

## Environment overview

The world is a directed supply-chain graph:

```text
farm -> processing -> warehouse -> retailer
```

Quick visual:

```text
Farm -> Processing Plant -> Warehouse -> Retail
```

Contamination usually starts upstream and becomes visible downstream later. That delay is what makes tracing difficult.

Each node stores:

- `node_id` and `node_type`
- hidden contamination level `true_level` in `[0.0, 1.0]`
- visible noisy `sensor_reading`
- `quarantined` flag
- `batch_ids` currently at the node
- downstream links

Contamination starts at one or more source farms and propagates downstream every step. The agent never sees the true contamination levels directly. It only sees noisy sensor readings, delayed retailer illness reports, and results from lab inspections it explicitly requests.

## What the agent sees

Every step returns a typed observation object with FoodCrisis fields plus a legacy compatibility shim.

Primary observation fields:

| Field                      | Meaning                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------------ |
| `timestep`                 | Current hour / step in the outbreak                                                              |
| `nodes`                    | Structured node list including sensor reading, quarantine state, batches, and downstream links   |
| `sensor_readings`          | Noisy contamination proxy by node                                                                |
| `illness_reports`          | Delayed retailer reports released so far                                                         |
| `quarantine_status`        | Current quarantine state per node                                                                |
| `lab_results`              | Exact contamination verdicts for completed inspections                                           |
| `traced_batches`           | Supply chain paths for batches that have been traced, mapping batch IDs to their origin and path |
| `lab_budget`               | Remaining lab tests                                                                              |
| `recall_budget`            | Remaining recall budget                                                                          |
| `public_trust`             | Starts near 1.0 and drops when the agent overreacts                                              |
| `natural_language_summary` | Judge-friendly and LLM-friendly summary of the current situation                                 |

The summary is the intended decision surface for language models. It highlights the top sensor spikes, recent illness reports, quarantine state, lab updates, traced batch paths, pending tests, remaining budgets, and public trust in one short paragraph.

## Action space

The environment accepts exact string actions:

| Action                 | Effect                                                                          |
| ---------------------- | ------------------------------------------------------------------------------- |
| `INSPECT <node_id>`    | Spend one lab token to get an exact contamination result on the next step       |
| `QUARANTINE <node_id>` | Block all outbound spread from that node                                        |
| `LIFT <node_id>`       | Remove quarantine and restore flow                                              |
| `RECALL <batch_id>`    | Remove a batch from the supply chain at recall-budget cost                      |
| `ALERT <node_id>`      | Issue a retailer warning that slows exposure but permanently reduces trust      |
| `TRACE <batch_id>`     | Trace a batch backward through the supply chain to identify its source and path |
| `WAIT`                 | Take no direct action and let the system evolve one step                        |

These are intentionally minimal. The difficulty comes from hidden state and delayed evidence, not from a large action vocabulary.

### About TRACE

`TRACE` is a low-cost information action that models real food-traceability workflows. When called on a batch, it returns the complete path the batch took through the supply chain and identifies the origin node. This allows agents to reason about supply structure and backwards causality without expensive lab tests.

- Cost: minimal (-0.1 reward)
- Benefit: eliminates uncertainty about batch origin and propagation path
- Use case: disambiguate between false signals and real source contamination
- Real-world mapping: corresponds to supplier records and batch traceability data under FSMA 204

TRACE results are stored in `traced_batches` and appear in the observation and `natural_language_summary`.

## Hidden dynamics

Each task runs the same core simulator:

- contamination starts at one or more hidden source farms
- contamination propagates downstream through weighted edges every step
- quarantines block outgoing spread
- alerts reduce exposure pressure for a few steps
- sensor readings are noisy and task-dependent
- retailer illness reports are delayed
- Task 3 can inject false sensor spikes and periodic re-seeding of source contamination

The agent must therefore learn containment, tracing, and triage under partial observability.

## Reward signal

The environment gives dense step-level reward, not just an end score. Rewards are strategically designed to teach outbreak _origin-finding_ rather than just symptom-chasing.

### Quarantine rewards (differentiated by contamination source):

- **Source quarantine**: +4.0 — Direct containment at origin
- **Non-source contaminated quarantine**: +2.0 — Downstream containment, less optimal
- **Wrong quarantine** (quarantining clean nodes): -2.0 — Precision penalty

### Recall rewards (differentiated by batch status):

- **Correct contaminated recall**: +1.5 — Removing actual contaminated batches
- **Wrong clean recall**: -1.0 — Penalty for recalling uncontaminated product

### Containment rewards:

- **Prevented contaminated shipments** (blocked by quarantine): +0.5 each — Direct signal for effective containment before illness arrives

### Operational penalties:

- **TRACE cost**: -0.1 — Small cost to encourage judicious information gathering
- **Urgency penalty**: -0.15 × active_uncontained_sources — Increases pressure as contamination spreads

### Why this structure improves learning:

The reward now explicitly distinguishes finding the _outbreak origin_ from simply chasing downstream symptoms. Quarantining a source farm (+4.0) yields twice the reward of quarantining a downstream processor (+2.0), so the agent gets a strong signal to trace backward rather than only reacting to highest sensor readings.

Recall is now symmetric: agents get positive feedback for removing contaminated product instead of only paying cost for recalls. This encourages disciplined, evidence-based recall decisions.

The prevented-shipment reward gives the agent a direct signal before illness damages arrive, which is critical in a delayed-observation environment — agents learn that _early containment_ prevents both contamination spread and delayed loss-of-life penalties.

This reward is meant to guide learning during an episode. Final leaderboard ranking is driven by the grader, not by raw summed reward alone.

## Grading

Each episode receives a deterministic final score in `[0.0, 1.0]` from four components:

- `containment`: how much contaminated product was prevented from reaching retail
- `precision`: how often the agent acted correctly instead of overreacting
- `speed`: how early the outbreak was contained
- `public_trust`: how much trust remained at episode end

Task-specific weights shift the emphasis from pure containment on easy episodes toward trust-preserving, budget-aware control on hard episodes.

## Task suite

| Task | Difficulty | Description                                                                    | Noise | Illness Delay | Lab Budget | Recall Budget | Max Steps |
| :--: | :--------: | :----------------------------------------------------------------------------- | ----: | :-----------: | :--------: | :-----------: | :-------: |
|  1   |    Easy    | Single source, low noise, fast reports, generous budgets                       |  0.05 |       1       |     10     |      100      |    48     |
|  2   |   Medium   | Multi-source outbreak with delayed reports and tighter budgets                 |  0.15 |       3       |     6      |      60       |    60     |
|  3   |    Hard    | Adversarial false spikes, delayed reports, re-seeding, and high trust pressure |  0.25 |       5       |     4      |      40       |    72     |

Difficulty progression is real:

- Task 1 can often be solved with TRACE → inspect-then-quarantine.
- Task 2 requires prioritization across multiple branches and distinguishing true sources from false spikes.
- Task 3 punishes naive "quarantine the highest sensor" strategies because some spikes are fake, sources can re-seed, and the reward structure explicitly rewards source-finding (+4.0) over downstream reactions (+2.0).

## Why the hard task is hard

Task 3 is designed to stress frontier agent behavior, not just basic tracing.

- Adversarial sensor noise can make clean nodes look dangerous.
- False signals force the agent to verify before overreacting.
- Re-seeding means source farms can become hazardous again, so one early good action is not enough.
- Trust is weighted more heavily, so blunt "quarantine everything" strategies score poorly.

This creates exactly the kind of ambiguity and delayed feedback that separates robust decision-making from brittle scripted behavior.

## Example episode

One strong FoodCrisis trajectory looks like this:

1. `TRACE farm_a_batch_001`
   - A batch with high downstream exposure appeared at the retailer. The agent traces it backward to identify the origin, confirming it originated from `farm_a`.
   - TRACE returns the full path: `farm_a -> processor_p1 -> warehouse_w1 -> retailer_r1`
2. `INSPECT farm_a`
   - Direct lab test to confirm the source is actually contaminated, not a false signal.
3. `QUARANTINE farm_a`
   - Lab confirms contamination. The agent quarantines the source farm (reward: +4.0), blocking all outbound propagation.
   - This is worth twice as much as quarantining downstream nodes, guiding the agent toward origin-finding.
4. `RECALL farm_a_batch_001`
   - The contaminated batch already reached the retailer. Recall it surgically (reward: +1.5 for correct recall).
5. `WAIT`
   - The agent waits one step to see if the quarantine prevents downstream illness, confirming containment success.

This is the pattern judges should expect: trace to understand, inspect to confirm, quarantine at the source, recall surgically, and avoid blind overreaction. The reward structure explicitly rewards finding the origin over symptom-chasing.

## Baselines and compatibility

This repository now includes a native deterministic baseline at [`baselines/food_crisis_agent.py`](./baselines/food_crisis_agent.py). It uses exact FoodCrisis actions and a simple strategy:

- inspect suspicious high-sensor nodes
- quarantine confirmed contaminated nodes
- lift quarantine from confirmed clean nodes
- recall batches from reported or confirmed affected nodes

Legacy compatibility is preserved:

- the package name remains `irce`
- the existing root `inference.py` still runs without changes
- legacy compatibility actions are mapped internally onto FoodCrisis actions so submission infrastructure does not break

That compatibility path is useful for packaging and deployment, but the native FoodCrisis baseline is the clearer reference agent for this benchmark.

Why the baseline is limited:

- it mostly follows local sensor evidence and does not employ TRACE for backward causality
- it uses shallow memory and no explicit backward tracing over the full graph
- it cannot reason strategically about misleading spikes the way a stronger LLM or learned policy can
- it does not distinguish between finding the outbreak origin (high reward) versus just reacting downstream (lower reward)

How an LLM or learned policy can do better:

- use TRACE to identify batch origin and supply chain path before committing to expensive quarantines
- combine illness reports with upstream graph structure and traced paths
- distinguish suspicious-but-unconfirmed nodes from confirmed sources
- _prioritize quarantining confirmed sources_ over reacting to high sensors, capturing the +4.0 reward signal
- spend lab budget where ambiguity is highest instead of where the sensor is merely largest
- preserve trust by avoiding unnecessary alerts and quarantines on nodes confirmed clean by TRACE

## Common failure modes

These are the mistakes that usually separate weak agents from strong ones:

- **Symptom-chasing without source-finding**: Quarantining high-sensor downstream nodes (+2.0) instead of tracing back to find the source (+4.0). This wastes budget and leaves the origin still spreading.
- **Over-quarantining without tracing**: Aggressively shutting down nodes based on sensor spikes alone without TRACE or INSPECT confirmation (wrong quarantine: -2.0, plus trust damage).
- **Ignoring illness signals**: Waiting too long to act after downstream retailer reports allows contaminated batches to keep moving and exposure to compound.
- **Wasting lab budget**: Spending inspections on already-clear nodes or obvious symptoms instead of high-ambiguity decision points leaves the agent blind when source identification is critical.
- **Overusing alerts**: Alerts buy time, but repeated warnings without traced or confirmed evidence make the final trust score collapse.
- **Not recalling contaminated batches**: Once a batch is confirmed contaminated and already downstream, recall (correct recall: +1.5) is more cost-effective than trying to contain at source retroactively.

Reference scores from `python baselines/food_crisis_agent.py` at `seed=7`:

| Policy                    |  Task 1 |  Task 2 |  Task 3 | Average |
| ------------------------- | ------: | ------: | ------: | ------: |
| `FoodCrisisBaselineAgent` | `0.494` | `0.590` | `0.642` | `0.575` |

These are reproducible sanity-check scores, not an optimized ceiling.

## Evaluation interpretation

As a rule of thumb:

- around `0.30-0.50`: the agent reacts, but containment is inconsistent or too costly
- around `0.50-0.70`: the agent is operationally useful and beats shallow heuristics
- around `0.70-0.85`: strong performance with good containment and disciplined budget use
- above `0.85`: very strong behavior with accurate containment, limited overreaction, and good trust preservation

A score near `0.50` usually means the agent solved part of the outbreak but wasted budget, acted too late, or damaged trust. A score near `0.80` means the agent contained most contamination while staying precise.

## LLM prompt template

The environment is already LLM-friendly because every step includes `natural_language_summary`. A good default prompt for agentic evaluation is:

System prompt:

```text
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
Valid formats:
- INSPECT <node_id>
- QUARANTINE <node_id>
- LIFT <node_id>
- RECALL <batch_id>
- ALERT <node_id>
- TRACE <batch_id>
- WAIT
```

User prompt:

```text
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

Strict output examples:

```text
TRACE farm_b_batch_001
INSPECT farm_b
QUARANTINE farm_b
RECALL farm_b_batch_004
ALERT retailer_r2
WAIT
```

### Deployment on HF Spaces

For HF Spaces deployment, this template integrates directly with the server running on port 7860. Users can submit observations from the `/state` endpoint to this prompt. See the [Docker](#docker) section below for quick setup.

## Quick start

### Install

```bash
pip install -e .
```

### Configure optional model access

`inference.py` supports OpenAI-compatible providers. If you do not configure a model, it falls back to a deterministic built-in policy and still prints scores.

```bash
cp .env.example .env
```

Example Hugging Face OpenAI-compatible setup:

```env
OPENAI_API_KEY=
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=deepseek-ai/DeepSeek-V3-0324
```

### Validate

```bash
openenv validate .
```

### Run inference

```bash
python inference.py
```

### Run the FoodCrisis baseline

```bash
python baselines/food_crisis_agent.py
```

### Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

### Docker

```bash
docker build -t foodcrisisenv:latest .

# Server runs on port 7860 (required for HF Spaces)
docker run --rm -p 7860:7860 --env-file .env foodcrisisenv:latest

# Example with Groq API:
docker run -d \
  -p 7860:7860 \
  -e HF_TOKEN=your_groq_api_key_here \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-8b-instant \
  --name foodcrisis-server \
  foodcrisisenv:latest

docker stop foodcrisis-server
```

**Port configuration**: The default port is 7860 (HF Spaces requirement). To use a different port in development, set `FOODCRISIS_PORT` environment variable:

```bash
docker run --rm -p 8080:8080 -e FOODCRISIS_PORT=8080 foodcrisisenv:latest
```

Test endpoints:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/state
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": {"action_type": "WAIT"}}'
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": 1}
```

## Project layout

```text
.
├── inference.py
├── train.py
├── openenv.yaml
├── Dockerfile
├── baselines/
│   └── food_crisis_agent.py
├── server/
│   └── app.py
├── src/irce/
│   ├── environment.py
│   ├── grading.py
│   ├── models.py
│   ├── rewards.py
│   ├── tasks.py
│   └── client.py
└── tests/
    └── test_environment.py
```

## License

MIT
