// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FoodCrisisEnv — Frontend Controller
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let currentTask = 1, lastState = null, episodeLog = [];
let llmMode = 'manual', autoPlayLLM = false;
let currentSessionId = "default"; // Auto-populated by /reset
const labBudgetMax = {1:10,2:6,3:4};
const recallBudgetMax = {1:100,2:60,3:40};
const maxSteps = {1:48,2:60,3:72};
const taskDescs = {
  1:"Single source, low noise, fast reports, generous budgets. Perfect for learning.",
  2:"Multi-source outbreak with delayed reports and tighter budgets.",
  3:"Adversarial false spikes, re-seeding, delayed reports, high trust pressure."
};

// ── Routing ──
function navigateTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  if (page === 'simulate') {
    document.getElementById('page-simulate').classList.add('active');
    window.location.hash = '#/simulate';
    if (!lastState) initSimulation();
  } else {
    document.getElementById('page-landing').classList.add('active');
    window.location.hash = '#/';
  }
}

function handleRoute() {
  const hash = window.location.hash;
  if (hash === '#/simulate') navigateTo('simulate');
  else navigateTo('landing');
}

// ── Init ──
window.addEventListener('load', () => { checkHealth(); handleRoute(); });
window.addEventListener('hashchange', handleRoute);

async function initSimulation() {
  try {
    const r = await fetch('/reset', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_id:1,seed:7})
    });
    const data = await r.json();
    updateUI(data.observation || data, null, null);
  } catch(e) { console.error('Init failed', e); }
}

async function checkHealth() {
  try {
    const r = await fetch('/health');
    if (r.ok) {
      document.getElementById('status-text').textContent = 'LIVE';
      document.getElementById('status-dot').style.background = 'var(--clean)';
    }
  } catch {
    document.getElementById('status-text').textContent = 'OFFLINE';
    document.getElementById('status-dot').style.background = 'var(--red)';
  }
}

// ── Mode toggle ──
function setMode(mode) {
  llmMode = mode === 'llm' ? 'llm-sample' : 'manual';
  autoPlayLLM = false;
  document.getElementById('mode-manual').classList.toggle('active', mode === 'manual');
  document.getElementById('mode-llm').classList.toggle('active', mode === 'llm');
  document.getElementById('manual-controls').style.display = mode === 'manual' ? 'block' : 'none';
  document.getElementById('llm-controls').style.display = mode === 'llm' ? 'block' : 'none';
}

function updateLLMMode() {
  const v = document.getElementById('llm-mode-select').value;
  llmMode = v;
  document.getElementById('llm-step-btn').style.display = v === 'llm-sample' ? 'block' : 'none';
  document.getElementById('llm-stop-btn').style.display = v === 'llm-auto' ? 'block' : 'none';
  if (v === 'llm-auto' && lastState && !lastState.done) {
    autoPlayLLM = true; runLLMStep();
  }
}

function stopAutoPlay() { autoPlayLLM = false; document.getElementById('llm-status').textContent = '⏹ Stopped'; }

// ── Task selection ──
function selectTask(t) {
  currentTask = t;
  document.querySelectorAll('.task-tab').forEach((el,i) => el.classList.toggle('active', i+1===t));
  document.getElementById('task-desc').textContent = taskDescs[t];
  document.getElementById('step-max').textContent = maxSteps[t];
}

// ── API calls ──
async function resetEnv() {
  episodeLog = []; updateLog();
  document.getElementById('done-overlay').classList.remove('show');
  try {
    const r = await fetch('/reset', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_id:currentTask, seed:7})
    });
    const data = await r.json();
    if (data.session_id) currentSessionId = data.session_id; // Capture session from openenv
    updateUI(data.observation || data, null, null);
  } catch(e) { console.error('Reset failed', e); }
}

async function fetchState() {
  document.getElementById('refresh-spinner').style.display = 'inline';
  try {
    const url = currentSessionId !== "default" ? `/state?session_id=${currentSessionId}` : '/state';
    const r = await fetch(url);
    const data = await r.json();
    updateUI(data.observation || data, null, null);
  } catch(e) { console.error('Fetch failed', e); }
  finally { document.getElementById('refresh-spinner').style.display = 'none'; }
}

async function sendAction(verb, targetType) {
  let target = '';
  if (targetType === 'node') {
    target = document.getElementById('target-node').value;
    if (!target) { showFlash('Select a target node first.', 'error'); return; }
  } else if (targetType === 'batch') {
    target = document.getElementById('target-batch').value;
    if (!target) { showFlash('Select a target batch first.', 'error'); return; }
  }
  const actionStr = target ? `${verb} ${target}` : verb;
  try {
    const payload = {action: {action_type: actionStr}};
    if (currentSessionId && currentSessionId !== "default") payload.session_id = currentSessionId;
    
    const r = await fetch('/step', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const data = await r.json();
    const obs = data.observation || data;
    const rew = data.reward ?? obs.reward ?? 0;
    const done = data.done ?? obs.done ?? false;
    updateUI(obs, rew, actionStr);
    addLogEntry(obs.step_count ?? episodeLog.length+1, actionStr, rew, obs.tool_result);
    if (done) { autoPlayLLM = false; showDone(obs); }
  } catch(e) { showFlash('Request failed: '+e.message, 'error'); }
}

// ── LLM ──
async function runLLMStep() {
  if (!lastState) return;
  const statusEl = document.getElementById('llm-status');
  statusEl.textContent = '⏳ LLM thinking...';
  try {
    const prompt = buildPrompt(lastState);
    statusEl.textContent = '📡 Querying LLM...';
    const llmResp = await queryLLM(prompt);
    const actionStr = parseLLMAction(llmResp);
    statusEl.textContent = '💡 ' + actionStr;
    const payload = {action: {action_type: actionStr}};
    if (currentSessionId && currentSessionId !== "default") payload.session_id = currentSessionId;
    
    const r = await fetch('/step', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const data = await r.json();
    const obs = data.observation || data;
    const rew = data.reward ?? obs.reward ?? 0;
    updateUI(obs, rew, actionStr);
    addLogEntry(obs.step_count ?? episodeLog.length+1, actionStr, rew, '');
    if (obs.done) { autoPlayLLM = false; statusEl.textContent = '✓ Complete'; showDone(obs); }
    else if (autoPlayLLM) setTimeout(runLLMStep, 1500);
  } catch(e) { statusEl.textContent = '❌ ' + e.message; autoPlayLLM = false; }
}

function buildPrompt(obs) {
  const summary = obs.natural_language_summary || '';
  const nodes = (obs.nodes||[]).map(n=>`${n.node_id} (${n.node_type})`).join(', ');
  const qNodes = Object.entries(obs.quarantine_status||{}).filter(([,v])=>v).map(([k])=>k);
  return `You are a food safety incident responder.\n\n${summary}\n\nState:\n- Step: ${obs.timestep||obs.step_count||0}\n- Nodes: ${nodes}\n- Quarantined: ${qNodes.join(', ')||'None'}\n- Lab budget: ${obs.lab_budget||0}\n- Recall budget: ${obs.recall_budget||0}\n- Trust: ${(obs.public_trust||1).toFixed(2)}\n\nActions: INSPECT <node>, QUARANTINE <node>, LIFT <node>, RECALL <batch>, TRACE <batch>, WAIT\n\nRespond with exactly ONE action string.`;
}

async function queryLLM(prompt) {
  const r = await fetch('/llm/decide', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:prompt})
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail||'LLM error'); }
  const data = await r.json();
  return data.action || 'WAIT';
}

function parseLLMAction(text) {
  const actions = ['INSPECT','QUARANTINE','LIFT','RECALL','TRACE','WAIT'];
  const upper = text.toUpperCase();
  for (const act of actions) {
    if (upper.includes(act)) {
      if (act==='WAIT') return 'WAIT';
      const m = text.match(new RegExp(act+'\\s+([\\w_]+)','i'));
      if (m) return `${act} ${m[1]}`;
      return act;
    }
  }
  return 'WAIT';
}

// ── UI Update ──
function updateUI(obs, reward, action) {
  if (!obs) return;
  lastState = obs;
  const step = obs.step_count ?? obs.timestep ?? 0;
  const mx = maxSteps[currentTask];
  const pct = Math.round((step/mx)*100);

  document.getElementById('step-cur').textContent = step;
  document.getElementById('progress-pct').textContent = pct+'%';
  const circ = 2*Math.PI*22;
  document.getElementById('progress-ring').style.strokeDashoffset = circ-(circ*pct/100);

  const lb = obs.lab_budget ?? 0, rb = obs.recall_budget ?? 0;
  const lbMax = labBudgetMax[currentTask], rbMax = recallBudgetMax[currentTask];
  document.getElementById('lab-budget').textContent = lb;
  document.getElementById('lab-budget').className = 'metric-val ' + (lb===0?'bad':lb<=2?'warn':'good');
  document.getElementById('lab-bar').style.width = Math.round((lb/lbMax)*100)+'%';
  document.getElementById('recall-budget').textContent = rb;
  document.getElementById('recall-budget').className = 'metric-val ' + (rb<10?'bad':rb<30?'warn':'');
  document.getElementById('recall-bar').style.width = Math.round((rb/rbMax)*100)+'%';

  const trust = obs.public_trust ?? 1.0;
  document.getElementById('trust-val').textContent = trust.toFixed(2);
  document.getElementById('trust-val').className = 'metric-val '+(trust<0.5?'bad':trust<0.75?'warn':'good');
  const tb = document.getElementById('trust-bar');
  tb.style.width = Math.round(trust*100)+'%';
  tb.style.background = trust<0.5?'var(--red)':trust<0.75?'var(--amber)':'var(--accent)';

  if (reward !== null && reward !== undefined) {
    const rEl = document.getElementById('last-reward');
    rEl.textContent = (reward>=0?'+':'')+reward.toFixed(2);
    rEl.style.color = reward>0?'var(--clean)':reward<0?'var(--red)':'var(--text3)';
  }
  if (obs.tool_result) {
    document.getElementById('tool-result').textContent = obs.tool_result;
    showFlash(obs.natural_language_summary ? obs.natural_language_summary.split('.')[0]+'.' : obs.tool_result,
      obs.tool_result==='SUCCESS'?'success':obs.tool_result==='ERROR'?'error':'ambig');
  }

  // Dropdowns
  const nodes = obs.nodes || [];
  const ns = document.getElementById('target-node'), bs = document.getElementById('target-batch');
  const pn = ns.value, pb = bs.value;
  ns.innerHTML = '<option value="">— select —</option>';
  bs.innerHTML = '<option value="">— select —</option>';
  const seen = new Set();
  nodes.forEach(n => {
    const o = document.createElement('option');
    o.value = n.node_id;
    const q = (obs.quarantine_status||{})[n.node_id]?' [Q]':'';
    const lr = (obs.lab_results||{})[n.node_id]?` [${obs.lab_results[n.node_id][0].toUpperCase()}]`:'';
    o.textContent = n.node_id+q+lr;
    if (o.value===pn) o.selected = true;
    ns.appendChild(o);
    (n.batch_ids||[]).forEach(bid => {
      if (!seen.has(bid)) { seen.add(bid);
        const bo = document.createElement('option'); bo.value=bid; bo.textContent=bid;
        if (bo.value===pb) bo.selected=true; bs.appendChild(bo);
      }
    });
  });

  // Lab pills
  const labRes = obs.lab_results || {};
  const lp = document.getElementById('lab-results-pills');
  lp.innerHTML = Object.keys(labRes).length
    ? Object.entries(labRes).map(([n,r])=>`<span class="pill ${r==='contaminated'?'pill-red':'pill-green'}">${n}: ${r}</span>`).join('')
    : '<span style="color:var(--text3);font-size:11px;">No results yet.</span>';

  // Quarantine pills
  const qStatus = obs.quarantine_status || {};
  const qNodes = Object.entries(qStatus).filter(([,v])=>v).map(([k])=>k);
  document.getElementById('quarantine-pills').innerHTML = qNodes.length
    ? qNodes.map(n=>`<span class="pill pill-amber">🚫 ${n}</span>`).join('')
    : '<span style="color:var(--text3);font-size:11px;">None.</span>';

  // Illness
  const illness = obs.illness_reports || [];
  document.getElementById('illness-reports').innerHTML = illness.length
    ? illness.map(r=>`<div class="illness-item"><span style="color:var(--amber);font-weight:600;">${r.retailer_id}</span> — <span style="color:var(--red)">${r.case_count} case(s)</span> <span style="color:var(--text3)">@ t${r.timestep_reported}</span></div>`).join('')
    : '<span style="color:var(--text3);font-size:11px;">No reports yet.</span>';

  // Traced
  const traced = obs.traced_batches || {};
  const tp = document.getElementById('traced-pills'), tpa = document.getElementById('trace-paths');
  if (!Object.keys(traced).length) { tp.innerHTML='<span style="color:var(--text3);font-size:11px;">None traced.</span>'; tpa.innerHTML=''; }
  else {
    tp.innerHTML = Object.keys(traced).map(b=>`<span class="pill pill-blue">🔍 ${b}</span>`).join('');
    tpa.innerHTML = Object.entries(traced).map(([b,p])=>`<div style="margin-top:4px;"><span style="color:var(--blue)">${b}:</span> <span style="color:var(--text3)">${Array.isArray(p)?p.join(' → '):p}</span></div>`).join('');
  }

  drawGraph(obs);
}

// ── Graph ──
function nodeColor(id, obs) {
  const lr = (obs.lab_results||{})[id];
  const s = (obs.sensor_readings||{})[id] ?? 0;
  if (lr==='contaminated') return {fill:'#2a0a0a',stroke:'var(--red)',glow:true};
  if (lr==='clean') return {fill:'#0a1f0a',stroke:'var(--clean)',glow:false};
  if (s>=0.7) return {fill:'#2a1a0a',stroke:'var(--amber)',glow:false};
  if (s>=0.4) return {fill:'#1a200a',stroke:'#84cc16',glow:false};
  return {fill:'#0c1219',stroke:'rgba(100,116,139,0.4)',glow:false};
}

function drawGraph(obs) {
  const svg = document.getElementById('supply-graph');
  const nodes = obs.nodes || [];
  if (!nodes.length) return;
  const byType = {farm:[],processing:[],warehouse:[],retailer:[]};
  nodes.forEach(n => { if(byType[n.node_type]) byType[n.node_type].push(n); });
  const W=800,H=480, cols=['farm','processing','warehouse','retailer'];
  const colX=[80,260,450,650], nodeW=120, nodeH=52, positions={};
  cols.forEach((type,ci) => {
    const g = byType[type], total=g.length;
    const span = Math.max(total*70,60), startY = H/2-span/2;
    g.forEach((n,i) => {
      positions[n.node_id] = {x:colX[ci]-nodeW/2, y:total===1 ? H/2-nodeH/2 : startY+i*(span/Math.max(total-1,1))-nodeH/2};
    });
  });
  let s = `<defs><marker id="arr" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="rgba(100,116,139,0.3)"/></marker><marker id="arr-red" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="var(--red)"/></marker><filter id="gf"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  const colLabels = ['FARMS','PROCESSING','WAREHOUSES','RETAILERS'];
  cols.forEach((t,ci) => { s+=`<text x="${colX[ci]}" y="22" text-anchor="middle" font-family="Inter" font-size="9" fill="rgba(100,116,139,0.3)" letter-spacing="2">${colLabels[ci]}</text>`; });
  const labRes = obs.lab_results||{}, qStatus = obs.quarantine_status||{};
  nodes.forEach(n => {
    const from = positions[n.node_id]; if(!from) return;
    (n.connected_to||[]).forEach(toId => {
      const to = positions[toId]; if(!to) return;
      const ic = labRes[n.node_id]==='contaminated', iq = qStatus[n.node_id];
      const x1=from.x+nodeW, y1=from.y+nodeH/2, x2=to.x, y2=to.y+nodeH/2, mx=(x1+x2)/2;
      const sc = ic&&!iq?'var(--red)':iq?'rgba(100,116,139,0.15)':'rgba(100,116,139,0.2)';
      const mk = ic&&!iq?'arr-red':'arr';
      const da = iq?'stroke-dasharray="4,4"':'';
      s+=`<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" stroke="${sc}" fill="none" stroke-width="${ic&&!iq?2.5:1.5}" ${da} marker-end="url(#${mk})" opacity="${iq?0.25:0.5}"/>`;
    });
  });
  nodes.forEach(n => {
    const pos = positions[n.node_id]; if(!pos) return;
    const {x,y} = pos, col = nodeColor(n.node_id, obs);
    const sensor = (obs.sensor_readings||{})[n.node_id]??0;
    const iq = qStatus[n.node_id], lr = labRes[n.node_id];
    const gf = col.glow?'filter="url(#gf)"':'';
    const label = n.node_id.length>14?n.node_id.slice(0,13)+'…':n.node_id;
    let icon=''; if(lr==='contaminated')icon='☣'; else if(lr==='clean')icon='✓'; else if(iq)icon='⚠'; else if(sensor>=0.7)icon='!';
    s+=`<g class="g-node" ${gf}><rect x="${x}" y="${y}" width="${nodeW}" height="${nodeH}" rx="8" fill="${col.fill}" stroke="${col.stroke}" stroke-width="${iq?2.5:1.5}" stroke-dasharray="${iq?'5,3':''}"/>`;
    if(icon) s+=`<text x="${x+nodeW-10}" y="${y+14}" text-anchor="middle" font-family="Inter" font-size="12" fill="${col.stroke}">${icon}</text>`;
    s+=`<text x="${x+nodeW/2}" y="${y+20}" text-anchor="middle" font-family="Inter" font-size="10" font-weight="700" fill="#e2e8f0">${label}</text>`;
    s+=`<text x="${x+nodeW/2}" y="${y+32}" text-anchor="middle" font-family="Inter" font-size="9" fill="${col.stroke}">${n.node_type.toUpperCase()}</text>`;
    s+=`<text x="${x+nodeW/2}" y="${y+46}" text-anchor="middle" font-family="JetBrains Mono" font-size="11" font-weight="700" fill="${sensor>=0.7?'var(--red)':sensor>=0.4?'var(--amber)':'rgba(100,116,139,0.5)'}">${sensor.toFixed(2)}</text>`;
    if((n.batch_ids||[]).length) s+=`<text x="${x+6}" y="${y+46}" font-family="Inter" font-size="8" fill="rgba(100,116,139,0.3)">${n.batch_ids.length}b</text>`;
    s+=`</g>`;
  });
  svg.innerHTML = s;
}

// ── Log ──
function addLogEntry(step,action,reward,result) { episodeLog.push({step,action,reward,result}); updateLog(); }
function updateLog() {
  const el = document.getElementById('episode-log');
  if (!episodeLog.length) { el.innerHTML='<span style="color:var(--text3)">Awaiting first action...</span>'; return; }
  el.innerHTML = [...episodeLog].reverse().slice(0,30).map(e => {
    const rc = (e.reward??0)>=0?'pos':'neg';
    const rs = e.reward!=null?((e.reward>=0?'+':'')+Number(e.reward).toFixed(2)):'—';
    return `<div class="log-entry"><span class="log-step">t${e.step}</span><span class="log-action">${e.action}</span><span class="log-reward ${rc}">${rs}</span></div>`;
  }).join('');
}

// ── Flash ──
function showFlash(msg,type) {
  const el = document.getElementById('result-flash');
  el.textContent = msg; el.className='result-flash '+type; el.style.display='block';
  clearTimeout(el._timer); el._timer=setTimeout(()=>{el.style.display='none';},4000);
}

// ── Done ──
function showDone(obs) {
  document.getElementById('done-overlay').classList.add('show');
  const p = obs.progress_hint ?? 0;
  const ds = document.getElementById('done-score');
  ds.textContent = p.toFixed(3);
  ds.style.color = p>=0.7?'var(--clean)':p>=0.5?'var(--amber)':'var(--red)';
}
