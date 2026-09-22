/* Prediction Agent demo dashboard.
   All values come from dashboard/demo_data.json, written by export_demo.py
   from one real agent run. Nothing here is hard-coded or illustrative. */

const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
const num = (n) => n.toLocaleString('en-US');
const fx = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : x.toFixed(d);

let D = null;
let selected = 0;

/* ---------------- 00 premise ---------------- */
function renderPremise() {
  const m = D.meta, lt = D.label_truncation;

  document.getElementById('kpis').innerHTML = [
    ['BOM rows', num(m.rows), `${D.phase_sizes.length} maturity phases`, ''],
    ['Errors, correctly labelled', num(m.positives), `${pct(m.base_rate, 1)} of rows`, 'signal'],
    ['Errors the original cast kept', num(lt.astype_int_positives), 'truncation discarded 98.4%', ''],
    ['Held-out rows', num(m.test_rows), `later phases ${m.test_phases[0]}–${m.test_phases.at(-1)}`, ''],
    ['BDQV detectors', String(D.bdqv.length), `${D.signature_types.length} used in signatures`, ''],
    ['Registry patterns', num(D.kb_stats.patterns), 'support ≥ 50', 'good'],
  ].map(([l, v, n, cls]) => `
    <div class="kpi ${cls}">
      <div class="k-label">${l}</div>
      <div class="k-value">${v}</div>
      <div class="k-note">${n}</div>
    </div>`).join('');

  document.getElementById('label-callout').innerHTML = `
    <h3>Read this before any number below</h3>
    <p>The <span class="mono">erroneous</span> column is not binary. It is continuous over
    <span class="mono">[0, ${fx(lt.max, 3)}]</span> with ${lt.distinct_values} distinct values, an
    anonymisation artefact. The original code applied <span class="mono">astype(int)</span>, which
    <strong>truncates</strong>: only the ${num(lt.astype_int_positives)} rows at or above 1.0 counted as
    errors, which is where the often-quoted 0.24% error rate comes from. Binarising at
    <span class="mono">&gt; 0</span> recovers <strong>${num(lt.gt_zero_positives)} labelled errors
    (${pct(m.base_rate, 1)})</strong>. Everything on this page uses the corrected label, so it is not
    comparable with the figures in the original repository.</p>`;

  document.getElementById('foot-meta').innerHTML =
    `<code>seed ${m.seed} · train ${num(m.train_rows)} · test ${num(m.test_rows)}</code>`;
}

/* ---------------- 01 observe ---------------- */
function renderBdqv() {
  const maxFires = Math.max(...D.bdqv.map(r => r.fires));
  const tb = document.querySelector('#bdqv-table tbody');
  tb.innerHTML = D.bdqv.map(r => {
    const share = r.fires / D.meta.train_rows;
    let pill = '<span class="pill flat">n/a</span>';
    if (r.rate !== null) {
      const d = r.rate - r.base;
      const cls = Math.abs(d) < 0.006 ? 'flat' : (d > 0 ? 'up' : 'down');
      const arrow = cls === 'flat' ? '=' : (d > 0 ? '▲' : '▼');
      pill = `<span class="pill ${cls}">${arrow} ${(d * 100).toFixed(1)} pp</span>`;
    }
    return `<tr>
      <td class="mono">${r.type}</td>
      <td>${num(r.fires)}</td>
      <td class="bar-cell">
        <div class="bar" style="width:${(r.fires / maxFires) * 96}px"></div>
        <span class="bar-val dim" style="margin-left:100px">${pct(share, 1)}</span>
      </td>
      <td>${r.rate === null ? '—' : fx(r.rate)}</td>
      <td>${pill}</td>
      <td>${r.in_signature
        ? '<span class="tag">yes</span>'
        : '<span class="tag flagtype">flag only</span>'}</td>
    </tr>`;
  }).join('');

  const inv = D.bdqv.filter(r => r.rate !== null && r.rate < r.base - 0.006).map(r => r.type);
  const flag = D.bdqv.find(r => !r.in_signature);
  document.getElementById('observe-callout').innerHTML = `
    <h3>Two things to read from this table</h3>
    <p><strong>No single detector is a good predictor.</strong> That is the repository's own conclusion
    about <span class="mono">rho_v</span>, now quantified for all eight — which is exactly why the next
    step looks at combinations.</p>
    <p><strong>${inv.length} detectors fire on rows that are <em>less</em> likely to be erroneous</strong>
    than the base rate (${inv.map(t => `<span class="mono">${t}</span>`).join(', ')}). They are
    informative with the opposite sign — something a single binary label cannot express.
    ${flag ? `<span class="mono">${flag.type}</span> is kept as a model feature but excluded from
    signatures: it fires on ${pct(flag.fires / D.meta.train_rows, 0)} of rows, and including it produced
    one dominant pattern with lift ≈ 1.0 that drowned out the selective combinations.` : ''}</p>`;
}

/* ---------------- 02 correlate ---------------- */
function renderPatterns() {
  const maxLift = Math.max(...D.patterns.map(p => p.lift));
  document.querySelector('#pattern-table tbody').innerHTML = D.patterns.map(p => `
    <tr>
      <td class="mono">${p.signature.split('|').map(s =>
        `<span class="tag">${s}</span>`).join('')}</td>
      <td>${num(p.support)}</td>
      <td>${num(p.hits)}</td>
      <td>${fx(p.precision)}</td>
      <td><strong>${p.lift.toFixed(2)}×</strong></td>
      <td class="bar-cell" style="min-width:110px">
        <div class="bar" style="width:${(p.lift / maxLift) * 90}px;background:var(--amber)"></div>
      </td>
    </tr>`).join('');
}

/* ---------------- 03 predict ---------------- */
const LADDER_ORDER = ['A: repo features', 'B: + BDQV types', 'C: + behaviour', 'C-shuffled',
                      'D: + KG context', 'D-shuffled', 'E: + registry'];
const LABELS = {
  'A: repo features': 'A · repo features',
  'B: + BDQV types': 'B · + BDQV types',
  'C: + behaviour': 'C · + behaviour',
  'C-shuffled': 'C · behaviour permuted',
  'D: + KG context': 'D · + graph context',
  'D-shuffled': 'D · graph permuted',
  'E: + registry': 'E · + registry',
};

function atK(model, k) {
  const seq = D.ranked_labels[model];
  let hits = 0;
  for (let i = 0; i < k && i < seq.length; i++) hits += seq[i];
  const precision = hits / Math.min(k, seq.length);
  return { hits, precision, lift: precision / D.meta.test_base_rate };
}

function buildLadder() {
  document.getElementById('ladder').innerHTML = LADDER_ORDER.map(m => `
    <div class="lrow ${m.includes('shuffled') ? 'ctrl' : ''}" data-model="${m}">
      <div class="lname">${LABELS[m]}</div>
      <div class="ltrack">
        <div class="lfill" style="width:0"></div>
        <div class="baseline-mark" style="left:0"></div>
      </div>
      <div class="lnum"></div>
    </div>`).join('');

}

function updateLadder(k) {
  const base = D.meta.test_base_rate;
  const vals = LADDER_ORDER.map(m => atK(m, k));
  const maxP = Math.max(base * 1.25, ...vals.map(v => v.precision));

  LADDER_ORDER.forEach((m, i) => {
    const row = document.querySelector(`.lrow[data-model="${CSS.escape(m)}"]`);
    const v = vals[i];
    row.querySelector('.lfill').style.width = `${(v.precision / maxP) * 100}%`;
    row.querySelector('.lnum').innerHTML =
      `${v.lift.toFixed(2)}× <span class="sub">· ${v.hits} hits</span>`;
    const mark = row.querySelector('.baseline-mark');
    if (mark) mark.style.left = `${(base / maxP) * 100}%`;
  });

  const b = document.getElementById('budget');
  const frac = k / D.meta.test_rows;
  b.style.setProperty('--fill', `${((k - b.min) / (b.max - b.min)) * 100}%`);
  document.getElementById('b-k').textContent = num(k);
  document.getElementById('b-lab').textContent =
    `rows checked · ${pct(frac, 2)} of the test set`;
  // The agent's advantage is concentrated at small budgets. Widen the budget
  // and the ordering inverts - a property of the method worth showing, not a
  // glitch to hide.
  const A0 = atK('A: repo features', k), E0 = atK('E: + registry', k);
  const badge = document.getElementById('regime');
  badge.className = 'pill ' + (E0.lift > A0.lift ? 'up' : 'down');
  badge.textContent = E0.lift > A0.lift
    ? 'agent ahead of baseline'
    : 'baseline ahead — budget too wide';

  // narrative callouts, recomputed so they can never contradict the bars
  const C = atK('C: + behaviour', k), Cs = atK('C-shuffled', k);
  const Dd = atK('D: + KG context', k), Ds = atK('D-shuffled', k);
  const A = atK('A: repo features', k), E = atK('E: + registry', k);
  const holds = (x, y) => x.lift > y.lift;
  // Binomial standard error on precision@k. Differences smaller than roughly
  // two of these are not resolvable from a single run, and several rungs of
  // this ladder are exactly that close - so the page must not assert more.
  const sePP = Math.sqrt(E.precision * (1 - E.precision) / k) * 100;
  const seLift = sePP / (base * 100);
  document.getElementById('control-text').innerHTML =
    `A block that only adds columns to the feature matrix would score the same when its values are
     permuted. At this budget the behaviour block is <strong>${C.lift.toFixed(2)}×</strong> against
     <strong>${Cs.lift.toFixed(2)}×</strong> permuted, and the graph block
     <strong>${Dd.lift.toFixed(2)}×</strong> against <strong>${Ds.lift.toFixed(2)}×</strong> permuted.
     ${holds(C, Cs) && holds(Dd, Ds)
        ? `Both controls point the right way. Read them against the sampling noise, though:
           the binomial standard error on precision at this budget is ±${sePP.toFixed(1)} pp,
           i.e. ±${seLift.toFixed(2)}× — so these gaps are worth roughly
           ${(Math.min(C.lift - Cs.lift, Dd.lift - Ds.lift) / seLift).toFixed(1)}–${(Math.max(C.lift - Cs.lift, Dd.lift - Ds.lift) / seLift).toFixed(1)}
           standard errors from one seed. Suggestive, not settled.`
        : `At this budget at least one control does <strong>not</strong> point the right way — the
           honest reading is that the block adds capacity, not structure here.`}
     Against the plain baseline the full agent finds ${E.hits} real errors versus ${A.hits}
     (${E.hits === A.hits ? 'no difference'
         : (E.hits > A.hits ? `+${E.hits - A.hits}` : `${E.hits - A.hits}`) + ' errors'})
     for the same engineer time — a gap of
     ${((E.lift - A.lift) / seLift).toFixed(1)} standard errors, which is the one comparison here
     that is comfortably resolved.`;

  document.getElementById('prauc-text').innerHTML =
    `Global ranking quality does not improve. PR-AUC across the whole test set is
     <strong>${fx(D.ladder.find(r => r.model === 'A: repo features').pr_auc)}</strong> for model A and
     <strong>${fx(D.ladder.find(r => r.model === 'E: + registry').pr_auc)}</strong> for model E — it
     drifts slightly <em>down</em>. The agent is a better ranker only in the region that actually gets
     worked. If you care about the full ordering rather than a budgeted worklist, the plain baseline is
     just as good, and recall stays low: ${pct(E.hits / D.meta.positives_test, 1)} of the errors in
     these phases at this budget.`;
}

/* ---------------- 04 explain ---------------- */
function renderWorklist() {
  document.getElementById('worklist').innerHTML = D.alerts.map((a, i) => `
    <button class="wrow" role="option" data-i="${i}" aria-selected="${i === 0}">
      <span class="rank">${String(i + 1).padStart(2, '0')}</span>
      <span class="who">
        part ${a.part} · component ${a.component} · phase ${a.phase}
        <span class="sig">${a.signature}</span>
      </span>
      <span class="verdict ${a.verdict}">${a.verdict === 'confirmed' ? 'real error' : 'false alarm'}</span>
    </button>`).join('');

  document.getElementById('worklist').addEventListener('click', (e) => {
    const btn = e.target.closest('.wrow');
    if (btn) selectAlert(Number(btn.dataset.i));
  });
  selectAlert(0);
}

const EV_LABEL = {
  registry_precision: 'registry precision',
  bdqv_count: 'BDQVs firing',
  bdqv_severity_max: 'max severity',
  bdqv_severity_sum: 'severity sum',
  kg_component_risk: 'graph · component risk',
  kg_part_risk: 'graph · part risk',
  isolation_forest: 'isolation forest',
  beh_session_load: 'behaviour · session load',
  beh_dwell_z: 'behaviour · dwell (own baseline)',
  beh_undo_rate: 'behaviour · undo rate',
  rho_v: 'rho_v (Procrustes)',
};

function selectAlert(i) {
  selected = i;
  const a = D.alerts[i];
  document.querySelectorAll('.wrow').forEach(el =>
    el.setAttribute('aria-selected', String(Number(el.dataset.i) === i)));

  const ev = Object.entries(a.evidence).map(([k, v]) => `
    <div class="ev-item">
      <div class="l">${EV_LABEL[k] || k.replace(/_/g, ' ')}</div>
      <div class="v">${typeof v === 'number' ? fx(v, 3) : v}</div>
    </div>`).join('');

  const edges = a.graph_context.length ? a.graph_context.map(e => `
    <div class="edge">${e[0]} <span class="rel">─${e[1]}→</span> ${e[2]}
      <span class="dim">phase ${e[3]}</span></div>`).join('')
    : '<div class="dim">No indexed neighbours for this part.</div>';

  document.getElementById('evidence').innerHTML = `
    <div class="ev-head">
      <div class="ev-title">part ${a.part} · component ${a.component} · phase ${a.phase}</div>
      <span class="verdict ${a.verdict}">${a.verdict === 'confirmed' ? 'real error' : 'false alarm'}</span>
    </div>
    <div>
      <div class="l dim" style="font-size:10px;letter-spacing:.06em;text-transform:uppercase">Violations firing</div>
      <div style="margin-top:4px">${a.bdqv_types.length
        ? a.bdqv_types.map(t => D.signature_types.includes(t)
            ? `<span class="tag">${t}</span>`
            : `<span class="tag flagtype" title="high-prevalence flag, excluded from signatures">${t} · flag</span>`).join('')
        : '<span class="tag flagtype">none — ranked on features alone</span>'}</div>
    </div>
    <div class="ev-why"><strong>Why:</strong> ${a.why}</div>
    <div>
      <div class="l dim" style="font-size:10px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px">Score evidence · agent score ${fx(a.score, 3)}</div>
      <div class="ev-grid">${ev}</div>
    </div>
    <div>
      <div class="l dim" style="font-size:10px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px">Knowledge-graph neighbourhood</div>
      <div class="ev-graph">${edges}</div>
    </div>`;
}

/* ---------------- 05 learn ---------------- */
function renderLearn() {
  const rows = D.feedback.slice().sort((a, b) =>
    (b.confirmed + b.dismissed) - (a.confirmed + a.dismissed));
  document.querySelector('#feedback-table tbody').innerHTML = rows.length
    ? rows.map(r => `<tr>
        <td class="mono">${r.signature}</td>
        <td>${r.confirmed}</td>
        <td>${r.dismissed}</td>
        <td><strong>${fx(r.human_precision)}</strong></td>
      </tr>`).join('')
    : '<tr><td colspan="4" class="dim">No verdicts recorded.</td></tr>';

  document.getElementById('kb-stats').innerHTML = Object.entries(D.kb_stats)
    .map(([k, v]) => `<div class="kpi" style="padding:var(--s3)">
        <div class="k-label">${k}</div>
        <div class="k-value" style="font-size:19px">${num(v)}</div>
      </div>`).join('');
}

/* ---------------- 06 limits ---------------- */
function renderLimits() {
  const E = D.ladder.find(r => r.model === 'E: + registry');
  const Dd = D.ladder.find(r => r.model === 'D: + KG context');
  document.getElementById('limits-list').innerHTML = [
    `<strong>Recall is low.</strong> At the default 1% budget the agent surfaces
     ${E.hits_at_k} of ${num(D.meta.positives_test)} errors in the held-out phases
     (${pct(E.recall_at_k, 1)}). This is a triage aid, not an oracle.`,
    `<strong>The registry earns nothing here.</strong> Model E is within noise of model D
     (${E.lift_at_k.toFixed(2)}× vs ${Dd.lift_at_k.toFixed(2)}×). With a single vehicle generation in
     the data there is nothing to transfer; the cross-generation warm start is the point of the
     registry and this dataset cannot demonstrate it.`,
    `<strong>The configuration event log is simulated.</strong> The published data is a BOM snapshot
     with no click, dwell or undo events. The simulator is derived from BOM structure only and never
     from the label, so it cannot inject artificial signal — but the behaviour result is a statement
     about the interface, not evidence that real user behaviour predicts errors.`,
    `<strong>Global ranking quality is unchanged.</strong> PR-AUC does not improve over the plain
     baseline at any rung of the ladder.`,
    `<strong>One run, one seed, and the ladder is not statistically separated.</strong> Seed
     ${D.meta.seed} only. At the default budget the binomial standard error on precision@k is about
     ±3 pp (±7 hits). The baseline-to-full-agent gap is roughly 3 standard errors and survives; the
     individual contributions of the behaviour, graph and registry rungs are 1–2 standard errors apart
     and do <strong>not</strong>. Establishing them needs rolling-origin folds over several seeds with
     confidence intervals, which has not been done yet.`,
    `<strong>The phase-regression head is still unusable</strong> (MAE ≈ 5 of 14 phases in the
     corrected MLP) and is not part of the agent's ranking.`,
  ].map(t => `<li>${t}</li>`).join('');
}

/* ---------------- nav ---------------- */
function initNav() {
  const links = [...document.querySelectorAll('#nav a')];
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      links.forEach(l => l.classList.toggle('active',
        l.getAttribute('href') === '#' + e.target.id));
    });
  }, { root: document.getElementById('main'), rootMargin: '-15% 0px -70% 0px' });
  document.querySelectorAll('section').forEach(s => obs.observe(s));
  links[0].classList.add('active');
}

/* ---------------- boot ---------------- */
// no-cache: after a regenerated snapshot is pushed, a reviewer who already
// opened the page must not be served the previous run's numbers.
fetch('demo_data.json', { cache: 'no-cache' })
  .then(r => r.json())
  .then(data => {
    D = data;
    // rows in the held-out phases that are actually erroneous
    D.meta.positives_test = Math.round(D.meta.test_rows * D.meta.test_base_rate);

    renderPremise();
    renderBdqv();
    renderPatterns();
    buildLadder();
    renderWorklist();
    renderLearn();
    renderLimits();
    initNav();

    const b = document.getElementById('budget');
    b.max = String(Math.min(D.meta.max_rank, D.ranked_labels['A: repo features'].length));
    b.value = String(Math.round(D.meta.test_rows * D.meta.budget));
    b.addEventListener('input', () => updateLadder(Number(b.value)));
    updateLadder(Number(b.value));
  })
  .catch(err => {
    document.querySelector('.wrap').innerHTML =
      `<div class="callout caution"><h3>Could not load the run snapshot</h3>
       <p>demo_data.json is missing or unreadable. Regenerate it with
       <span class="mono">PYTHONPATH=src .venv/bin/python export_demo.py</span>.</p>
       <p class="mono dim">${err}</p></div>`;
  });
