/* Study tools: spaced-repetition review + progress / mastery tracking + weak-spots quiz. Fully offline.
 *
 * Everything here reuses content generated in advance (the 4 347 page questions and 1 065 concepts); nothing
 * is generated at runtime. State is saved to localStorage and, through serve.py, to data/user/study.json.
 *
 * State
 *   pages[n]   = {secs, status:'understood'|'confusing'|null, deck:true|false|undefined, u}
 *   cards[id]  = {due, ivl, ease, reps, lapses, last, u}       id = "q:<page>:<qid>" | "c:<conceptId>"
 *   answers[k] = {r, w, last, u}                               k  = "q:<page>:<qid>"
 *   days[date] = {reviews, fresh, secs}
 *   log        = [[time, cardId, rating], ...]   (last 5 000 reviews)
 */
(() => {
  'use strict';
  const LS_KEY = 'bc:study';
  const DAY = 864e5, STUDIED_SECS = 45, TICK = 5;
  const ELIGIBLE = new Set(['content', 'figure_only']);
  const S0 = () => ({ version: 1, pages: {}, cards: {}, answers: {}, days: {}, log: [], settings: { newPerDay: 20 } });
  let D = S0();
  const X = { server: false, saveTimer: null, questions: null, byPage: new Map(), concepts: null, conceptsByPage: new Map(), lastInput: Date.now(), bar: null };
  const BC = () => window.BC;
  const h = (...a) => BC().el(...a);
  const $ = (s, r = document) => r.querySelector(s);
  const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; };
  const dayRec = () => (D.days[today()] ||= { reviews: 0, fresh: 0, secs: 0 });
  const meta = () => BC().S.meta;
  const kindOf = (n) => meta().page_map[n]?.kind;
  const fmtDur = (s) => s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
  const fmtIvl = (d) => d < 1 / 24 ? `${Math.max(1, Math.round(d * 1440))} min` : d < 1 ? `${Math.round(d * 24)} h` : d < 30 ? `${Math.round(d)} d` : d < 365 ? `${(d / 30).toFixed(1)} mo` : `${(d / 365).toFixed(1)} y`;
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  // ------------------------------------------------------------------ persistence (newest entry wins)
  function merge(into, src) {
    if (!src || typeof src !== 'object') return;
    for (const k of ['pages', 'cards', 'answers']) for (const [id, v] of Object.entries(src[k] || {})) { const cur = into[k][id]; if (!cur || (v?.u || 0) > (cur.u || 0)) into[k][id] = v; }
    for (const [d, v] of Object.entries(src.days || {})) { const cur = into.days[d] ||= { reviews: 0, fresh: 0, secs: 0 }; for (const f of ['reviews', 'fresh', 'secs']) cur[f] = Math.max(cur[f] || 0, v?.[f] || 0); }
    const seen = new Set(into.log.map((l) => l[0] + l[1])); for (const l of src.log || []) if (Array.isArray(l) && !seen.has(l[0] + l[1])) into.log.push(l);
    into.log.sort((a, b) => a[0] - b[0]); if (into.log.length > 5000) into.log = into.log.slice(-5000);
    if (src.settings) into.settings = { ...into.settings, ...src.settings };
  }
  function save(now = false) {
    try { localStorage.setItem(LS_KEY, JSON.stringify(D)); } catch { /* full / blocked */ }
    clearTimeout(X.saveTimer); X.saveTimer = setTimeout(push, now ? 50 : 1500);
  }
  async function push() {
    if (!X.server) return;
    try { const r = await fetch('api/study', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...D, saved_at: new Date().toISOString() }) }); if (!r.ok) throw new Error(r.status); }
    catch { BC().toast('Could not write study progress to disk — it is kept in this browser.'); }
  }
  async function load() {
    try { merge(D, JSON.parse(localStorage.getItem(LS_KEY) || 'null')); } catch { /* ignore */ }
    try { const r = await fetch('api/study', { cache: 'no-store' }); if (r.ok) { X.server = true; merge(D, await r.json()); } } catch { /* static server */ }
    save();
  }
  const touch = (o) => { o.u = Date.now(); return o; };

  // ------------------------------------------------------------------ content (questions + concepts, loaded once)
  async function ensureContent() {
    if (X.questions) return;
    const idx = await BC().fetchJSON('data/quizzes/index.json');
    const files = await Promise.all(idx.chapters.map((c) => BC().fetchJSON('data/quizzes/' + c.file).catch(() => ({ questions: [] }))));
    X.questions = files.flatMap((f) => f.questions || []).filter((q) => q.page && ELIGIBLE.has(kindOf(q.page)) && q.type !== 'self');
    X.questions.forEach((q) => { q.key = `q:${q.page}:${q.id}`; if (!X.byPage.has(q.page)) X.byPage.set(q.page, []); X.byPage.get(q.page).push(q); });
    X.concepts = (await BC().fetchJSON('data/concepts/concepts.json')).map((c) => ({ ...c, key: 'c:' + c.id, home: (c.pages || []).find((p) => ELIGIBLE.has(kindOf(p))) })).filter((c) => c.home && c.definition);
    X.concepts.forEach((c) => { if (!X.conceptsByPage.has(c.home)) X.conceptsByPage.set(c.home, []); X.conceptsByPage.get(c.home).push(c); });
  }

  // ------------------------------------------------------------------ page progress
  const pg = (n) => D.pages[n] || { secs: 0, status: null };
  const studied = (n) => { const p = pg(n); return p.secs >= STUDIED_SECS || !!p.status; };
  const inDeck = (n) => ELIGIBLE.has(kindOf(n)) && (pg(n).deck === true || (pg(n).deck !== false && studied(n)));
  function setPage(n, patch) { D.pages[n] = touch({ ...pg(n), ...patch }); save(); refreshBar(); updateBadge(); }
  function tick() {
    if (document.visibilityState !== 'visible' || Date.now() - X.lastInput > 120e3 || !$('#modal').hidden) return;
    const n = BC().S.page; if (!n) return;
    const before = studied(n);
    const p = pg(n); D.pages[n] = touch({ ...p, secs: (p.secs || 0) + TICK }); dayRec().secs += TICK;
    if (!before && studied(n)) { refreshBar(); updateBadge(); if (ELIGIBLE.has(kindOf(n))) BC().toast(`p.${n} added to your review deck`); }
    else if (D.pages[n].secs % 30 === 0) refreshBar();
    try { localStorage.setItem(LS_KEY, JSON.stringify(D)); } catch {}
    if (D.pages[n].secs % 60 === 0) save();
  }

  // ------------------------------------------------------------------ spaced repetition (SM-2 style)
  function schedule(st, rating, now = Date.now()) {
    const s = st ? { ...st } : { ivl: 0, ease: 2.5, reps: 0, lapses: 0 };
    if (rating === 1) { if (st && st.reps > 0) s.lapses++; s.reps = 0; s.ivl = 0; s.ease = Math.max(1.3, s.ease - 0.2); s.due = now + 10 * 60e3; }
    else {
      const good = s.reps === 0 ? 1 : s.reps === 1 ? 3 : s.ivl * s.ease;
      let ivl = rating === 2 ? (s.reps === 0 ? 0.5 : Math.max(1, s.ivl * 1.2)) : rating === 3 ? good : Math.max(good + 1, s.reps === 0 ? 4 : s.ivl * s.ease * 1.3);
      if (rating === 2) s.ease = Math.max(1.3, s.ease - 0.15); if (rating === 4) s.ease += 0.15;
      if (ivl > 3) ivl *= 0.95 + Math.random() * 0.1;
      s.ivl = Math.round(ivl * 100) / 100; s.reps++; s.due = now + s.ivl * DAY;
    }
    s.last = now; return touch(s);
  }
  function eligibleCards() {
    const out = [];
    for (let n = 1; n <= BC().S.pageCount; n++) {
      if (!inDeck(n)) continue;
      (X.byPage.get(n) || []).forEach((q) => out.push({ id: q.key, kind: 'q', page: n, q }));
      (X.conceptsByPage.get(n) || []).forEach((c) => out.push({ id: c.key, kind: 'c', page: n, c }));
    }
    return out;
  }
  function queue() {
    const now = Date.now(); const all = eligibleCards();
    const due = all.filter((c) => D.cards[c.id] && D.cards[c.id].due <= now).sort((a, b) => D.cards[a.id].due - D.cards[b.id].due);
    const left = Math.max(0, (D.settings.newPerDay || 20) - (dayRec().fresh || 0));
    const fresh = all.filter((c) => !D.cards[c.id]).slice(0, left);
    return { due, fresh, all };
  }
  function counts() {
    if (!X.questions) return null;
    const { due, fresh, all } = queue(); let learning = 0, mature = 0;
    all.forEach((c) => { const st = D.cards[c.id]; if (!st) return; if (st.ivl >= 21) mature++; else learning++; });
    return { due: due.length, fresh: fresh.length, total: all.length, unseen: all.filter((c) => !D.cards[c.id]).length, learning, mature };
  }
  function updateBadge() { const b = $('#btn-review'); const c = counts(); if (b && c) { const n = c.due + c.fresh; b.textContent = n ? `Review (${n})` : 'Review ✓'; b.classList.toggle('due', n > 0); } }
  function recordAnswer(key, ok) {
    if (!key) return; const a = D.answers[key] || { r: 0, w: 0 };
    D.answers[key] = touch({ ...a, r: a.r + (ok ? 1 : 0), w: a.w + (ok ? 0 : 1), last: Date.now() }); save();
  }
  const streak = () => { let n = 0; const d = new Date(); if (!(D.days[today()]?.reviews)) d.setDate(d.getDate() - 1);
    for (;;) { const k = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; if (D.days[k]?.reviews) { n++; d.setDate(d.getDate() - 1); } else return n; } };

  async function openReview() {
    const bc = BC(); const body = h('div', { class: 'review' }); bc.openModal(body);
    body.append(h('p', { class: 'meta' }, 'Loading your review deck…'));
    await ensureContent();
    const { due, fresh } = queue();
    const session = [...due.slice(0, 200)];
    fresh.forEach((c, i) => session.splice(Math.min(session.length, 3 * i + 2), 0, c));   // interleave new cards
    let done = 0, again = 0;
    const head = h('div', { class: 'review-head' }); const stage = h('div', { class: 'review-stage' });
    const newSel = h('select', { title: 'New cards per day' }, ...[10, 20, 40, 80].map((n) => h('option', { value: n }, `${n} new / day`)));
    newSel.value = D.settings.newPerDay || 20; newSel.addEventListener('change', () => { D.settings.newPerDay = +newSel.value; save(); openReview(); });
    const addCh = h('button', { onclick: () => { addChapter(meta().page_map[bc.S.page]?.chapter); openReview(); } }, 'Add current chapter to review');
    body.replaceChildren(h('h2', {}, 'Spaced-repetition review'), head, h('div', { class: 'filters' }, newSel, addCh, h('span', { class: 'meta' }, 'Keys: Space = show answer · 1 Again · 2 Hard · 3 Good · 4 Easy')), stage);
    const setHead = () => { head.textContent = `${Math.max(0, session.length - done)} card(s) left in this session · reviewed today: ${dayRec().reviews} · streak: ${streak()} day(s)`; };
    const next = () => {
      setHead(); stage.replaceChildren();
      if (done >= session.length) {
        const c = counts(); const upcoming = Object.values(D.cards).map((s) => s.due).filter((t) => t > Date.now()).sort((a, b) => a - b)[0];
        stage.append(h('div', { class: 'review-done' }, h('h3', {}, session.length ? 'Session complete' : 'Nothing to review right now'),
          h('p', {}, session.length ? `You reviewed ${session.length} card(s)${again ? `, ${again} marked “Again” will come back in 10 minutes` : ''}.` : (c.total ? 'All cards are scheduled for later.' : 'Your deck is empty: pages join it automatically after about 45 seconds of study, when you mark them, or with “Add current chapter to review”.')),
          upcoming ? h('p', { class: 'meta' }, `Next card due ${new Date(upcoming).toLocaleString()}.`) : '',
          h('p', { class: 'meta' }, `Deck: ${c.total} cards · ${c.unseen} not yet seen · ${c.learning} learning · ${c.mature} mature (interval ≥ 21 days).`)));
        updateBadge(); return;
      }
      const card = session[done]; const st = D.cards[card.id]; const pm = meta().page_map[card.page] || {};
      const box = h('div', { class: 'flashcard' });
      const info = h('div', { class: 'meta' }, st ? `review · interval ${fmtIvl(st.ivl || 0)}` : 'new card', ' · ', pm.chapter || '', ' · ', bc.pageLink(card.page, `PDF p.${card.page}`));
      const back = h('div', { class: 'card-back', hidden: true });
      const rate = h('div', { class: 'rate', hidden: true });
      let suggested = 3, revealed = false;
      const reveal = (sugg) => { if (revealed) return; revealed = true; if (sugg) suggested = sugg; back.hidden = false; rate.hidden = false; showBtn.remove(); $$('.rate button', box).forEach((b) => b.classList.toggle('sugg', +b.dataset.r === suggested)); bc.renderMath(back); };
      const showBtn = h('button', { class: 'primary', onclick: () => reveal() }, 'Show answer (Space)');
      if (card.kind === 'q') {
        const q = card.q;
        box.append(info, h('div', { class: 'qtype' }, { short: 'short answer', conceptual: 'conceptual', math: 'mathematical', mcq: 'multiple choice' }[q.type] || q.type), h('div', { class: 'md front', html: bc.md(q.question) }));
        if (q.type === 'mcq' && q.options?.length) {
          const opts = h('div', { class: 'opts' });
          q.options.forEach((o) => { const lab = h('label', {}, h('input', { type: 'radio', name: 'rv' + done }), ' ', h('span', { html: bc.md(o).replace(/^<p>|<\/p>$/g, '') }));
            lab.addEventListener('click', () => { if (revealed) return; const ok = o === q.answer; lab.classList.add(ok ? 'correct' : 'wrong'); [...opts.children].find((l) => l._o === q.answer)?.classList.add('correct'); recordAnswer(q.key, ok); card.answered = true; reveal(ok ? 3 : 1); });
            lab._o = o; opts.append(lab); });
          box.append(opts);
        } else box.append(showBtn);
        back.append(h('div', {}, h('b', {}, 'Answer: '), h('span', { html: bc.md(q.answer).replace(/^<p>|<\/p>$/g, '') })), h('div', { class: 'md', html: bc.md(q.explanation) }));
      } else {
        const c = card.c;
        box.append(info, h('div', { class: 'qtype' }, 'concept'), h('div', { class: 'front' }, 'Explain in your own words: ', h('b', {}, c.name)), showBtn);
        back.append(h('div', { class: 'md', html: bc.md(c.definition) }), h('div', { class: 'meta' }, 'Appears on: ', ...(c.pages || []).filter((p) => ELIGIBLE.has(kindOf(p))).slice(0, 12).map((p) => [bc.pageLink(p), ' '])));
      }
      [[1, 'Again'], [2, 'Hard'], [3, 'Good'], [4, 'Easy']].forEach(([r, t]) => {
        const pv = schedule(st, r); const lbl = r === 1 ? '10 min' : fmtIvl(pv.ivl);
        rate.append(h('button', { 'data-r': r, onclick: () => grade(r) }, h('b', {}, t), h('small', {}, lbl)));
      });
      const grade = (r) => {
        D.cards[card.id] = schedule(D.cards[card.id], r); D.log.push([Date.now(), card.id, r]); if (D.log.length > 5000) D.log.shift();
        const day = dayRec(); day.reviews++; if (!st) day.fresh++;
        if (card.kind === 'q' && !card.answered) recordAnswer(card.q.key, r >= 2);
        if (r === 1) { again++; session.push(card); }
        save(); done++; next();
      };
      box.append(back, rate); stage.append(box); bc.renderMath(box);
      X.keyHandler = (e) => { if ($('#modal').hidden || !stage.isConnected) return; if (e.target.matches('textarea, select, input[type=search], input[type=text]')) return;
        if (e.key === ' ') { e.preventDefault(); reveal(); } else if (revealed && ['1', '2', '3', '4'].includes(e.key)) { e.preventDefault(); grade(+e.key); } };
    };
    if (!X.keyBound) { document.addEventListener('keydown', (e) => X.keyHandler && X.keyHandler(e)); X.keyBound = true; }
    next();
  }
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ------------------------------------------------------------------ weak-spots quiz (interleaved across chapters)
  function accuracy(keys) { let r = 0, w = 0; keys.forEach((k) => { const a = D.answers[k]; if (a) { r += a.r; w += a.w; } }); return r + w ? { r, w, acc: r / (r + w) } : null; }
  async function weakQuiz(area, count, scoreEl) {
    await ensureContent();
    const pool = X.questions.filter((q) => studied(q.page) || D.answers[q.key]);
    if (!pool.length) { area.replaceChildren(h('p', {}, 'Study a few pages first (or answer some questions): the weak-spots quiz draws from pages you have studied, weighted toward what you got wrong or marked confusing.')); return; }
    const weight = (q) => { const a = D.answers[q.key]; const st = D.cards[q.key]; const p = pg(q.page); const pa = accuracy((X.byPage.get(q.page) || []).map((x) => x.key));
      return 1 + (a ? 4 * a.w / (a.r + a.w) : 1) + (p.status === 'confusing' ? 3 : 0) + (st ? Math.min(3, st.lapses) : 0) + (pa && pa.acc < 0.6 ? 2 : 0) + (a && Date.now() - a.last < 3600e3 ? -1 : 0); };
    const picked = []; const cand = pool.map((q) => ({ q, w: Math.max(0.2, weight(q)) }));
    while (picked.length < count && cand.length) { const tot = cand.reduce((s, c) => s + c.w, 0); let x = Math.random() * tot, i = 0; while ((x -= cand[i].w) > 0 && i < cand.length - 1) i++; picked.push(cand.splice(i, 1)[0].q); }
    // interleave: avoid two consecutive questions from the same chapter where possible
    const ch = (q) => meta().page_map[q.page]?.chapter; const out = []; const rest = shuffle(picked);
    while (rest.length) { const i = rest.findIndex((q) => !out.length || ch(q) !== ch(out[out.length - 1])); out.push(rest.splice(i < 0 ? 0 : i, 1)[0]); }
    let right = 0, answered = 0; area.replaceChildren(h('p', { class: 'meta' }, `${out.length} questions from ${new Set(out.map(ch)).size} chapter(s), weighted toward your weak spots.`));
    out.forEach((q, i) => { const c = BC().questionCard({ ...q }, true); c.prepend(h('div', { class: 'meta' }, `Question ${i + 1} of ${out.length} · ${ch(q)}`));
      c.addEventListener('answered', (e) => { answered++; if (e.detail?.ok) right++; if (scoreEl) scoreEl.textContent = `Score: ${right}/${answered}`; }); area.append(c); });
  }

  // ------------------------------------------------------------------ chapter helpers
  const chapterPages = (title) => { const c = meta().chapters.find((x) => x.title === title); if (!c) return []; const out = []; for (let n = c.start; n <= c.end; n++) out.push(n); return out; };
  function addChapter(title) {
    const ps = chapterPages(title).filter((n) => ELIGIBLE.has(kindOf(n))); if (!ps.length) { BC().toast('This chapter has no study pages'); return; }
    ps.forEach((n) => { D.pages[n] = touch({ ...pg(n), deck: true }); }); save(); updateBadge(); refreshBar(); BC().toast(`Added ${ps.length} pages of “${title}” to your review deck`);
  }

  // ------------------------------------------------------------------ progress dashboard
  async function openProgress() {
    const bc = BC(); await ensureContent();
    const c = counts(); const m = meta();
    const studyPages = Object.keys(m.page_map).map(Number).filter((n) => ELIGIBLE.has(kindOf(n)));
    const studiedN = studyPages.filter(studied).length, und = studyPages.filter((n) => pg(n).status === 'understood').length, conf = studyPages.filter((n) => pg(n).status === 'confusing');
    const totalSecs = Object.values(D.pages).reduce((s, p) => s + (p.secs || 0), 0);
    const acc = accuracy(Object.keys(D.answers));
    const tile = (big, small) => h('div', { class: 'tile' }, h('b', {}, big), h('span', {}, small));
    const tiles = h('div', { class: 'tiles' },
      tile(`${studiedN} / ${studyPages.length}`, 'pages studied'), tile(String(und), 'marked understood'), tile(String(conf.length), 'marked confusing'),
      tile(fmtDur(totalSecs), 'study time'), tile(acc ? `${Math.round(acc.acc * 100)}%` : '–', `question accuracy${acc ? ` (${acc.r + acc.w} answers)` : ''}`),
      tile(String(c.due + c.fresh), 'cards to review now'), tile(`${streak()} d`, 'review streak'), tile(`${c.mature} / ${c.total}`, 'mature cards'));
    // page map
    const legend = h('div', { class: 'legend' }, ...[['', 'not opened'], ['seen', 'opened'], ['studied', 'studied'], ['understood', 'understood'], ['confusing', 'confusing']].map(([k, t]) => h('span', {}, h('i', { class: 'sq ' + k }), t)));
    const mapBox = h('div', { class: 'pagemap' });
    m.chapters.filter((ch) => ch.number).forEach((ch) => {
      const row = h('div', { class: 'pm-row' }, h('div', { class: 'pm-label', title: ch.title }, ch.title));
      const sq = h('div', { class: 'pm-squares' });
      for (let n = ch.start; n <= ch.end; n++) { const p = pg(n); const cls = p.status || (studied(n) ? 'studied' : p.secs ? 'seen' : ''); sq.append(h('a', { class: 'sq ' + cls, 'data-page': n, href: '#', title: `PDF p.${n} · ${fmtDur(p.secs || 0)}${p.status ? ' · ' + p.status : ''}`, onclick: (e) => { e.preventDefault(); bc.closeModal(); bc.gotoPage(n); } })); }
      row.append(sq); mapBox.append(row);
    });
    // chapter table
    const tbl = h('table', { class: 'chapter-table' }, h('tr', {}, ...['Chapter', 'Studied', 'Understood', 'Confusing', 'Accuracy', 'Cards due', ''].map((t) => h('th', {}, t))));
    m.chapters.filter((ch) => ch.number).forEach((ch) => {
      const ps = chapterPages(ch.title).filter((n) => ELIGIBLE.has(kindOf(n)));
      const s = ps.filter(studied).length, u = ps.filter((n) => pg(n).status === 'understood').length, cf = ps.filter((n) => pg(n).status === 'confusing').length;
      const a = accuracy(ps.flatMap((n) => (X.byPage.get(n) || []).map((q) => q.key)));
      const dueN = ps.reduce((t, n) => t + [...(X.byPage.get(n) || []), ...(X.conceptsByPage.get(n) || [])].filter((x) => D.cards[x.key] && D.cards[x.key].due <= Date.now()).length, 0);
      tbl.append(h('tr', {}, h('td', {}, h('a', { class: 'pageref', 'data-page': ch.start }, ch.title)),
        h('td', {}, h('div', { class: 'bar' }, h('i', { style: `width:${Math.round(100 * s / (ps.length || 1))}%` })), h('small', {}, `${s}/${ps.length}`)),
        h('td', {}, String(u)), h('td', {}, cf ? String(cf) : ''), h('td', { class: a && a.acc < 0.6 ? 'weak' : '' }, a ? `${Math.round(a.acc * 100)}% (${a.r + a.w})` : '–'), h('td', {}, dueN ? String(dueN) : ''),
        h('td', {}, h('button', { onclick: () => { addChapter(ch.title); openProgress(); } }, 'Add to review'))));
    });
    // weak + confusing lists
    const weakPages = studyPages.map((n) => ({ n, a: accuracy((X.byPage.get(n) || []).map((q) => q.key)) })).filter((x) => x.a && x.a.r + x.a.w >= 2 && x.a.acc < 0.7).sort((p, q) => p.a.acc - q.a.acc).slice(0, 25);
    const pageItem = (n, extra) => h('li', {}, bc.pageLink(n, `PDF p.${n}`), ' — ', m.page_map[n]?.section || m.page_map[n]?.chapter || '', extra ? h('span', { class: 'meta' }, ' · ' + extra) : '');
    const quizArea = h('div'); const scoreEl = h('div', { class: 'score' });
    const qn = h('select', {}, ...[10, 20, 40].map((n) => h('option', { value: n }, `${n} questions`)));
    const exportBtn = h('button', { onclick: () => { const url = URL.createObjectURL(new Blob([JSON.stringify(D, null, 1)], { type: 'application/json' })); const a = h('a', { href: url, download: `dlwp-study-progress-${today()}.json` }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000); } }, 'Export progress');
    bc.openModal(h('div', { class: 'progress' },
      h('h2', {}, 'Your progress'), h('p', { class: 'meta' }, X.server ? 'Saved automatically to data/user/study.json and to this browser.' : 'Saved in this browser only (start the app with serve.py to also save to data/user/study.json).', ' ', exportBtn),
      tiles,
      h('h3', {}, 'Weak-spots quiz'), h('p', { class: 'meta' }, 'Mixed questions from the pages you studied, weighted toward questions you got wrong, cards you forgot, low-accuracy pages and pages you marked confusing. Chapters are interleaved, which strengthens recall.'),
      h('div', { class: 'quiz-opts' }, qn, h('button', { class: 'primary', onclick: () => weakQuiz(quizArea, +qn.value, scoreEl) }, 'Start weak-spots quiz'), h('button', { onclick: () => { bc.closeModal(); openReview(); } }, 'Open review')), scoreEl, quizArea,
      h('h3', {}, 'Pages map'), legend, mapBox,
      h('h3', {}, 'By chapter'), tbl,
      h('div', { class: 'two-col' },
        h('div', {}, h('h3', {}, `Pages you marked confusing (${conf.length})`), conf.length ? h('ul', {}, ...conf.map((n) => pageItem(n, fmtDur(pg(n).secs || 0)))) : h('p', { class: 'meta' }, 'None yet — use “? Confusing” at the top of a page’s notes.')),
        h('div', {}, h('h3', {}, 'Weakest pages by quiz accuracy'), weakPages.length ? h('ul', {}, ...weakPages.map((x) => pageItem(x.n, `${Math.round(x.a.acc * 100)}% of ${x.a.r + x.a.w}`))) : h('p', { class: 'meta' }, 'Answer some questions (quiz, review or the page notes) to see this.')))));
  }

  // ------------------------------------------------------------------ status bar at the top of each page's notes
  function pageBar(n) {
    const bar = h('div', { class: 'study-bar' }); bar.dataset.page = n; X.bar = bar; fillBar(bar); return bar;
  }
  function fillBar(bar) {
    const n = +bar.dataset.page; const p = pg(n); const eligible = ELIGIBLE.has(kindOf(n));
    const mk = (st, label, title) => h('button', { class: 'st ' + st + (p.status === st ? ' on' : ''), title, onclick: () => setPage(n, { status: p.status === st ? null : st }) }, label);
    const cards = X.questions ? (X.byPage.get(n) || []).length + (X.conceptsByPage.get(n) || []).length : null;
    bar.replaceChildren(mk('understood', '✓ Understood', 'Mark this page as understood'), mk('confusing', '? Confusing', 'Mark this page as confusing — it will be weighted in the weak-spots quiz'),
      h('span', { class: 'meta' }, `studied ${fmtDur(p.secs || 0)}`, eligible ? (inDeck(n) ? ` · in review deck${cards !== null ? ` (${cards} cards)` : ''}` : ` · joins review after ${STUDIED_SECS} s`) : ''),
      eligible && !inDeck(n) ? h('button', { class: 'st', onclick: () => setPage(n, { deck: true }) }, '+ Review now') : '',
      eligible && inDeck(n) && pg(n).deck !== false ? h('button', { class: 'st subtle', title: 'Remove this page from the review deck', onclick: () => setPage(n, { deck: false }) }, '− review') : '');
  }
  function refreshBar() { if (X.bar && X.bar.isConnected) fillBar(X.bar); }

  // ------------------------------------------------------------------ public API
  window.Study = {
    openReview, openProgress, weakQuiz, pageBar, recordAnswer, addChapter, _state: () => D, _ensure: ensureContent,
    async init() {
      await load();
      ['mousemove', 'keydown', 'wheel', 'pointerdown', 'touchstart'].forEach((ev) => document.addEventListener(ev, () => { X.lastInput = Date.now(); }, { passive: true, capture: true }));
      document.addEventListener('scroll', () => { X.lastInput = Date.now(); }, { passive: true, capture: true });
      document.addEventListener('answered', (e) => { const q = e.detail?.q; if (q && q.page && q.id && !e.target.closest?.('.review')) recordAnswer(`q:${q.page}:${q.id}`, !!e.detail.ok); });
      document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') save(true); });
      setInterval(tick, TICK * 1000);
      ensureContent().then(() => { updateBadge(); refreshBar(); }).catch((e) => console.warn('study content', e));
      setInterval(updateBadge, 60e3);
    },
  };
})();
