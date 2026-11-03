/* Book companion — offline application core. No network, no AI at runtime. */
(() => {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') n.className = v; else if (k === 'html') n.innerHTML = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else if (v !== null && v !== undefined) n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k !== null && k !== undefined) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  };
  const store = {
    get: (k, d) => { try { const v = localStorage.getItem('bc:' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set: (k, v) => { try { localStorage.setItem('bc:' + k, JSON.stringify(v)); } catch {} },
  };
  const toast = (msg, ms = 2200) => { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(t._t); t._t = setTimeout(() => t.hidden = true, ms); };
  const fetchJSON = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };
  const pad3 = (n) => String(n).padStart(3, '0');
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ---------------------------------------------------------------- state
  const S = { meta: null, page: 1, pageCount: 0, sync: true, scale: 1, fit: 'width', pkg: null, pkgPage: 0, pdf: null,
    pageSizes: {}, rendered: new Map(), textCache: new Map(), search: null, searchDocs: null, matches: [], matchIdx: -1, thumbsOn: false };

  // ---------------------------------------------------------------- markdown & math
  marked.setOptions({ breaks: false, gfm: true });
  const PAGE_REF = /\b(?:PDF\s+)?p(?:p|age|ages)?\.?\s?(\d{1,3})(?:\s?[–-]\s?(\d{1,3}))?\b/g;
  function linkPageRefs(html) {
    // turn "p.61", "page 61", "pp. 61–63" into links (PDF numbering, as the generator was told to use)
    return html.replace(/(>[^<]*)/g, (seg) => seg.replace(PAGE_REF, (m, a, b) => {
      const n = +a; if (!(n >= 1 && n <= S.pageCount)) return m;
      return `<a class="pageref" data-page="${n}">${m}</a>`;
    }));
  }
  function md(text) {
    if (!text) return '';
    // protect math from markdown
    const holes = []; const prot = String(text).replace(/\$\$[\s\S]+?\$\$|\$[^\n$]+?\$/g, (m) => { holes.push(m); return `\u0000${holes.length - 1}\u0000`; });
    let html = DOMPurify.sanitize(marked.parse(prot));
    html = html.replace(/\u0000(\d+)\u0000/g, (_, i) => esc(holes[+i]));
    return linkPageRefs(html);
  }
  function renderMath(root) {
    try { renderMathInElement(root, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false }, { left: '\\[', right: '\\]', display: true }], throwOnError: false }); } catch (e) { console.warn(e); }
  }
  function katexBlock(latex) {
    const d = el('div', { class: 'math-latex' });
    try { katex.render(latex, d, { displayMode: true, throwOnError: false }); } catch (e) { d.textContent = latex; }
    return d;
  }
  const srcTag = (s) => s ? el('span', { class: `src src-${s}`, title: { book: 'Directly supported by the book', ai_explanation: 'Additional explanation added by the AI', analogy: 'Illustrative analogy / example (AI)', inference: 'Interpretation / inference (AI)' }[s] || s }, { book: 'book', ai_explanation: 'AI explanation', analogy: 'analogy', inference: 'inference' }[s] || s) : '';

  // ---------------------------------------------------------------- notes panel
  const SECTIONS = [
    ['overview', 'Overview'], ['summary', 'Summary'], ['paragraphs', 'Paragraphs'], ['concepts', 'Concepts'], ['math', 'Math'],
    ['code', 'Code'], ['visualizations', 'Visuals'], ['deeper', 'Deeper'], ['misconceptions', 'Misconceptions'],
    ['connections', 'Connections'], ['takeaways', 'Takeaways'], ['questions', 'Questions'],
  ];
  const openState = store.get('open', {});
  function section(id, title, count, body, defaultOpen = true) {
    const d = el('details', { class: 'sec', id: `sec-${id}` });
    if (openState[id] ?? defaultOpen) d.open = true;
    d.append(el('summary', {}, title, count !== null ? el('span', { class: 'count' }, `(${count})`) : ''));
    const b = el('div', { class: 'sec-body' }); b.append(...[body].flat()); d.append(b);
    d.addEventListener('toggle', () => { openState[id] = d.open; store.set('open', openState); });
    return d;
  }
  const mdDiv = (t, cls = 'md') => el('div', { class: cls, html: md(t) });
  const pageLink = (n, text) => el('a', { class: 'pageref', 'data-page': n }, text || `p.${n}`);

  async function loadNotes(n) {
    if (S.pkgPage === n) return;
    S.pkgPage = n;
    const box = $('#notes');
    const info = S.meta.page_map[n] || {};
    let pkg = null;
    if (info.has_notes !== false) { try { pkg = await fetchJSON(`data/pages/page_${pad3(n)}.json`); } catch (e) { pkg = null; } }
    if (S.pkgPage !== n) return; // superseded
    S.pkg = pkg;
    box.replaceChildren();
    const head = el('div', { class: 'page-title' });
    head.append(el('h1', {}, pkg ? pkg.overview.title : (info.section || info.chapter || `Page ${n}`)));
    head.append(el('div', { class: 'crumbs' }, el('span', {}, info.chapter || ''), info.section ? el('span', {}, info.section) : '', info.subsection ? el('span', {}, info.subsection) : '',
      el('span', {}, `PDF p.${n}`, info.printed ? ` · printed p.${info.printed}` : '')));
    box.append(head);
    if (!pkg) {
      box.append(el('div', { class: 'placeholder' }, el('p', {}, 'No educational package has been generated for this page yet.'),
        el('p', {}, `Run the generation stage for page ${n} (see README) — the pipeline is resumable.`)));
      buildNav(null); return;
    }
    const g = pkg.generation;
    const badges = el('div', {}, g.status === 'light' ? el('span', { class: 'badge' }, 'light page') : '',
      g.confidence !== 'high' ? el('span', { class: `badge ${g.confidence === 'low' ? 'low' : 'warn'}` }, `confidence: ${g.confidence}`) : '',
      ...(g.review_flags || []).map((f) => el('span', { class: 'badge warn', title: f }, '⚑ review: ' + f.slice(0, 60))));
    head.append(badges);
    const o = pkg.overview;
    box.append(section('overview', 'Page overview', null, [
      el('p', {}, el('b', {}, 'Main topic: '), o.main_topic),
      el('p', {}, el('b', {}, 'Central idea: '), mdDiv(o.central_idea, 'md inline')),
      el('p', {}, el('b', {}, 'Why it matters: '), mdDiv(o.importance, 'md inline')),
      o.learning_objectives?.length ? el('div', {}, el('h5', {}, 'Learning objectives'), el('ul', {}, ...o.learning_objectives.map((x) => el('li', { html: md(x) })))) : '',
      o.prerequisites?.length ? el('div', {}, el('h5', {}, 'Prerequisite knowledge'), el('ul', {}, ...o.prerequisites.map((x) => el('li', { html: md(x) })))) : '',
    ]));
    box.append(section('summary', 'Detailed summary', null, mdDiv(pkg.summary)));
    if (pkg.paragraphs?.length) box.append(section('paragraphs', 'Paragraph-by-paragraph', pkg.paragraphs.length, pkg.paragraphs.map((p, i) => el('div', { class: 'item', 'data-ref': p.ref },
      el('div', { class: 'excerpt', title: 'Click to show this passage in the book', onclick: () => locateInBook(p.excerpt) }, `“${p.excerpt}…”`),
      el('h5', {}, 'What it says'), mdDiv(p.what_it_says), el('h5', {}, 'What it means'), mdDiv(p.what_it_means),
      el('h5', {}, 'Why it matters'), mdDiv(p.why_it_matters),
      p.technical_idea ? [el('h5', {}, 'Underlying technical idea'), mdDiv(p.technical_idea)] : '',
      p.terminology?.length ? [el('h5', {}, 'Terminology'), el('div', { class: 'terms' }, ...p.terminology.map((t) => el('span', {}, el('b', {}, t.term + ': '), el('span', { html: md(t.meaning).replace(/^<p>|<\/p>$/g, '') }))))] : '',
      p.connections?.length ? [el('h5', {}, 'Connections'), el('ul', { class: 'conn-list' }, ...p.connections.map((c) => el('li', { html: md(c) })))] : '')), false));
    if (pkg.concepts?.length) box.append(section('concepts', 'Important concepts', pkg.concepts.length, pkg.concepts.map((c) => el('div', { class: 'item', id: `concept-${c.id}` },
      el('h4', {}, c.name, srcTag(c.source), el('a', { class: 'badge', href: '#', title: 'Open in knowledge graph', onclick: (e) => { e.preventDefault(); openGraph(c.id); } }, 'graph')),
      el('h5', {}, 'Definition'), mdDiv(c.definition), el('h5', {}, 'Intuition'), mdDiv(c.intuition), el('h5', {}, 'Technical explanation'), mdDiv(c.technical),
      el('h5', {}, 'Practical significance'), mdDiv(c.significance), el('h5', {}, 'Example'), mdDiv(c.example), el('h5', {}, 'Common misunderstanding'), mdDiv(c.misconception),
      c.related?.length ? [el('h5', {}, 'Related concepts'), el('div', { class: 'terms' }, ...c.related.map((r) => el('a', { class: 'pageref', href: '#', onclick: (e) => { e.preventDefault(); openGraph(r); } }, r.replace(/_/g, ' '), ' ')))] : ''))));
    if (pkg.math?.length) box.append(section('math', 'Mathematical explanations', pkg.math.length, pkg.math.map((m) => el('div', { class: 'item', id: `math-${m.id}` },
      el('h4', {}, m.title, srcTag(m.source), m.book_ref ? el('span', { class: 'badge', html: linkPageRefs(esc(m.book_ref)) }) : ''),
      katexBlock(m.latex), m.source_text ? el('p', { class: 'excerpt' }, 'In the book: ', el('code', {}, m.source_text)) : '',
      el('h5', {}, 'Variables'), el('table', { class: 'var-table' }, ...m.variables.map((v) => el('tr', {}, el('td', { html: md('$' + v.symbol.replace(/^\$|\$$/g, '') + '$') }), el('td', { html: md(v.meaning) })))),
      el('h5', {}, 'Operations'), mdDiv(m.operations), el('h5', {}, 'Why this formula exists'), mdDiv(m.purpose), el('h5', {}, 'Interpretation'), mdDiv(m.interpretation),
      el('h5', {}, 'Intuition'), mdDiv(m.intuition), m.numerical_example ? [el('h5', {}, 'Numerical example'), mdDiv(m.numerical_example)] : ''))));
    if (pkg.code?.length) box.append(section('code', 'Code explanations', pkg.code.length, pkg.code.map((c) => el('div', { class: 'item', id: `code-${c.id}` },
      el('h4', {}, c.listing ? `Listing ${c.listing} — ` : '', c.title),
      el('div', { class: 'code-orig' }, el('h5', {}, 'Original code (from the book)'), el('pre', {}, el('code', {}, c.original_code))),
      el('h5', {}, 'What it does'), mdDiv(c.what_it_does),
      c.line_explanations?.length ? [el('h5', {}, 'Line by line'), el('table', { class: 'lines' }, ...c.line_explanations.map((l) => el('tr', {}, el('td', {}, l.lines), el('td', { html: md(l.explanation) }))))] : '',
      el('h5', {}, 'Inputs and outputs'), mdDiv(c.inputs_outputs), el('h5', {}, 'Underlying algorithm'), mdDiv(c.algorithm), el('h5', {}, 'Relationship to the theory'), mdDiv(c.theory_link),
      c.parameters?.length ? [el('h5', {}, 'Important parameters'), el('ul', {}, ...c.parameters.map((p) => el('li', {}, el('code', {}, p.name), ' — ', el('span', { html: md(p.role).replace(/^<p>|<\/p>$/g, '') }))))] : '',
      c.pitfalls?.length ? [el('h5', {}, 'Common errors & misconceptions'), el('ul', {}, ...c.pitfalls.map((p) => el('li', { html: md(p) })))] : '',
      c.illustrative_code ? el('div', { class: 'code-illus' }, el('h5', {}, 'Illustrative code (written by the AI, not from the book)', srcTag('ai_explanation')), el('pre', {}, el('code', {}, c.illustrative_code))) : ''))));
    if (pkg.visualizations?.length) {
      const vizBodies = pkg.visualizations.map((v) => { const d = el('div', { class: 'viz', id: `viz-${v.id}` }, el('h4', {}, v.title), el('p', { class: 'why' }, v.why)); d._viz = v; return d; });
      box.append(section('visualizations', 'Diagrams & interactive visualizations', pkg.visualizations.length, vizBodies));
      vizBodies.forEach((d) => renderViz(d._viz, d));
    }
    if (pkg.deeper?.length) box.append(section('deeper', 'Going deeper', pkg.deeper.length, pkg.deeper.map((d) => el('div', { class: 'item' }, el('h4', {}, d.title, el('span', { class: 'badge' }, d.kind.replace('_', ' ')), srcTag(d.source)), mdDiv(d.body)))));
    if (pkg.misconceptions?.length) box.append(section('misconceptions', 'Common misconceptions', pkg.misconceptions.length, pkg.misconceptions.map((m) => el('div', { class: 'item mis' }, el('div', { class: 'claim', html: md(m.claim).replace(/^<p>|<\/p>$/g, '') }), el('div', { class: 'corr', html: md(m.correction).replace(/^<p>|<\/p>$/g, '') })))));
    const C = pkg.connections || {}; const connKeys = [['prerequisites', 'Prerequisites'], ['earlier', 'Introduced earlier'], ['later', 'Developed later'], ['related_math', 'Related math'], ['related_code', 'Related code'], ['related_figures', 'Related figures']];
    const connCount = connKeys.reduce((a, [k]) => a + (C[k]?.length || 0), 0);
    if (connCount) box.append(section('connections', 'Connections to other pages', connCount, connKeys.filter(([k]) => C[k]?.length).map(([k, t]) => el('div', {}, el('h5', {}, t), el('ul', { class: 'conn-list' }, ...C[k].map((l) => el('li', {}, pageLink(l.page, `p.${l.page}${S.meta.page_map[l.page]?.printed ? ' (printed ' + S.meta.page_map[l.page].printed + ')' : ''}`), ' — ', el('span', { html: md(l.note).replace(/^<p>|<\/p>$/g, '') }))))))));
    if (pkg.takeaways?.length) box.append(section('takeaways', 'Key takeaways', pkg.takeaways.length, el('ul', { class: 'takeaways' }, ...pkg.takeaways.map((t) => el('li', { html: md(t) })))));
    if (pkg.questions?.length) box.append(section('questions', 'Review questions', pkg.questions.length, pkg.questions.map((q) => questionCard(q)), false));
    renderMath(box);
    buildNav(pkg);
    applyNoteSearch();
    box.scrollTop = 0;
  }
  function questionCard(q, showPage = false) {
    const card = el('div', { class: 'question' }, el('div', { class: 'qtype' }, { short: 'short answer', conceptual: 'conceptual', math: 'mathematical', mcq: 'multiple choice', self: 'self-assessment' }[q.type] || q.type, showPage && q.page_refs?.[0] ? [' · ', pageLink(q.page_refs[0])] : ''), el('div', { class: 'md', html: md(q.question) }));
    const ans = el('div', { class: 'answer', hidden: true }, el('b', {}, 'Answer: '), el('span', { html: md(q.answer).replace(/^<p>|<\/p>$/g, '') }), el('div', { class: 'md', html: md(q.explanation) }), q.page_refs?.length ? el('div', {}, 'See: ', ...q.page_refs.map((p) => [pageLink(p), ' '])) : '');
    if (q.type === 'mcq' && q.options?.length) {
      const opts = el('div', { class: 'opts' });
      q.options.forEach((o) => { const lab = el('label', {}, el('input', { type: 'radio', name: 'q-' + q.id + Math.random() }), ' ', el('span', { html: md(o).replace(/^<p>|<\/p>$/g, '') }));
        lab.addEventListener('click', () => { $$('label', opts).forEach((l) => l.classList.remove('correct', 'wrong')); lab.classList.add(o === q.answer ? 'correct' : 'wrong'); $$('label', opts).find((l) => l._o === q.answer)?.classList.add('correct'); ans.hidden = false; card._res = o === q.answer; card.dispatchEvent(new CustomEvent('answered', { bubbles: true, detail: o === q.answer })); }); lab._o = o; opts.append(lab); });
      card.append(opts);
    } else card.append(el('button', { onclick: (e) => { ans.hidden = !ans.hidden; e.target.textContent = ans.hidden ? 'Show answer' : 'Hide answer'; } }, 'Show answer'));
    card.append(ans); renderMath(card); return card;
  }
  function buildNav(pkg) {
    const nav = $('#note-nav'); nav.replaceChildren();
    SECTIONS.forEach(([id, t]) => { const has = !!$(`#sec-${id}`); nav.append(el('a', { href: `#sec-${id}`, class: has ? '' : 'empty', onclick: (e) => { e.preventDefault(); const s = $(`#sec-${id}`); if (!s) return; s.open = true; s.scrollIntoView({ behavior: 'smooth', block: 'start' }); } }, t)); });
  }
  function applyNoteSearch() {
    const q = $('#note-search').value.trim().toLowerCase();
    const box = $('#notes');
    $$('mark.ns', box).forEach((m) => m.replaceWith(m.textContent));
    box.normalize();
    $$('.item, .question, .viz', box).forEach((it) => it.classList.remove('hidden-by-search'));
    if (!q) return;
    $$('.item, .question, .viz', box).forEach((it) => { if (!it.textContent.toLowerCase().includes(q)) it.classList.add('hidden-by-search'); });
    const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT, { acceptNode: (n) => n.parentElement.closest('.hidden-by-search, .katex, pre, script') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    const nodes = []; while (walker.nextNode()) if (walker.currentNode.textContent.toLowerCase().includes(q)) nodes.push(walker.currentNode);
    nodes.forEach((n) => { const parts = n.textContent.split(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'ig')); const frag = document.createDocumentFragment(); parts.forEach((p, i) => frag.append(i % 2 ? el('mark', { class: 'ns' }, p) : p)); n.replaceWith(frag); });
    $$('details.sec', box).forEach((d) => { if ($('mark.ns', d)) d.open = true; });
  }

  // ---------------------------------------------------------------- visualizations
  let mermaidReady = false;
  function renderViz(v, box) {
    try {
      if (v.type === 'widget') {
        const w = el('div', { class: 'widget' }); box.append(w);
        if (!window.Widgets || !window.Widgets[v.widget]) throw new Error(`unknown widget "${v.widget}"`);
        window.Widgets[v.widget](w, v.params || {}, { el, md, katexBlock, renderMath, gotoPage });
      } else if (v.type === 'mermaid') {
        if (!mermaidReady) { mermaid.initialize({ startOnLoad: false, theme: isDark() ? 'dark' : 'default', securityLevel: 'strict', fontFamily: 'inherit' }); mermaidReady = true; }
        const d = el('div', { class: 'mermaid' }); box.append(d);
        mermaid.render('mm' + Math.random().toString(36).slice(2), v.mermaid).then(({ svg }) => { d.innerHTML = DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true }, ADD_TAGS: ['foreignObject'], ADD_ATTR: ['dominant-baseline'] }); }).catch((e) => { d.append(el('div', { class: 'viz-error' }, 'Diagram failed to render: ' + e.message)); d.append(el('pre', {}, v.mermaid)); });
      } else if (v.type === 'chart') {
        const c = el('canvas'); box.append(c);
        const cfg = JSON.parse(JSON.stringify(v.chart)); cfg.options = cfg.options || {}; cfg.options.responsive = true; cfg.options.maintainAspectRatio = false;
        cfg.options.plugins = cfg.options.plugins || {}; Chart.defaults.color = getComputedStyle(document.body).color; Chart.defaults.borderColor = getComputedStyle(document.body).getPropertyValue('--line');
        box.style.height = ''; c.parentElement.style.minHeight = '260px'; c.style.height = '280px';
        new Chart(c, cfg);
      } else if (v.type === 'svg') {
        box.append(el('div', { html: DOMPurify.sanitize(v.svg, { USE_PROFILES: { svg: true } }) }));
      }
    } catch (e) { box.append(el('div', { class: 'viz-error' }, 'Visualization error: ' + e.message)); }
  }
  const isDark = () => document.documentElement.dataset.theme === 'dark' || (document.documentElement.dataset.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);

  // ---------------------------------------------------------------- PDF viewer
  const viewer = $('#viewer'), pagesBox = $('#pages');
  let io, activeIO, visible = new Map();
  async function initPDF() {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'assets/pdfjs/pdf.worker.min.js';
    S.pdf = await pdfjsLib.getDocument({ url: 'book/original_book.pdf', cMapPacked: true }).promise;
    S.pageCount = S.pdf.numPages;
    $('#page-total').textContent = '/ ' + S.pageCount; $('#page-input').max = S.pageCount;
    const p1 = await S.pdf.getPage(1); const vp = p1.getViewport({ scale: 1 });
    S.base = { w: vp.width, h: vp.height };
    for (let i = 1; i <= S.pageCount; i++) {
      const d = el('div', { class: 'pdfpage', 'data-page': i }, el('span', { class: 'num' }, `PDF ${i}${S.meta.page_map[i]?.printed ? ' · ' + S.meta.page_map[i].printed : ''}`));
      pagesBox.append(d);
    }
    io = new IntersectionObserver((ents) => ents.forEach((e) => { const n = +e.target.dataset.page; if (e.isIntersecting) renderPage(n); else unrenderPage(n); }), { root: viewer, rootMargin: '1200px 0px' });
    activeIO = new IntersectionObserver((ents) => { ents.forEach((e) => visible.set(+e.target.dataset.page, e.intersectionRatio)); updateActive(); }, { root: viewer, threshold: [0, .1, .25, .5, .75, 1] });
    $$('.pdfpage').forEach((d) => { io.observe(d); activeIO.observe(d); });
    applyZoom();
  }
  function scaleFor() {
    const w = viewer.clientWidth - 40, h = viewer.clientHeight - 30;
    if (S.fit === 'width') return w / S.base.w;
    if (S.fit === 'page') return Math.min(w / S.base.w, h / S.base.h);
    return S.scale;
  }
  function applyZoom(keepPage = true) {
    const cur = S.page;
    const sc = scaleFor(); S.scale = sc; S.zoomPx = sc * (window.devicePixelRatio || 1);
    $('#zoom-label').textContent = Math.round(sc * 100) + '%';
    $$('.pdfpage').forEach((d) => { const n = +d.dataset.page; const sz = S.pageSizes[n] || S.base; d.style.width = Math.floor(sz.w * sc) + 'px'; d.style.height = Math.floor(sz.h * sc) + 'px'; });
    S.rendered.forEach((_, n) => { unrenderPage(n, true); });
    if (keepPage) scrollToPage(cur, false);
    $$('.pdfpage').forEach((d) => { const r = d.getBoundingClientRect(), v = viewer.getBoundingClientRect(); if (r.bottom > v.top - 1200 && r.top < v.bottom + 1200) renderPage(+d.dataset.page); });
  }
  async function renderPage(n) {
    const d = $(`.pdfpage[data-page="${n}"]`); if (!d || S.rendered.get(n) === S.scale) return;
    S.rendered.set(n, S.scale);
    const page = await S.pdf.getPage(n);
    if (S.rendered.get(n) !== S.scale) return;
    const vp = page.getViewport({ scale: S.scale }); const dpr = window.devicePixelRatio || 1;
    S.pageSizes[n] = { w: vp.width / S.scale, h: vp.height / S.scale };
    const canvas = el('canvas'); canvas.width = Math.floor(vp.width * dpr); canvas.height = Math.floor(vp.height * dpr);
    canvas.style.width = Math.floor(vp.width) + 'px'; canvas.style.height = Math.floor(vp.height) + 'px';
    d.style.width = canvas.style.width; d.style.height = canvas.style.height;
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    const task = page.render({ canvasContext: ctx, viewport: vp }); d._task = task;
    try { await task.promise; } catch { return; }
    if (S.rendered.get(n) !== S.scale) return;
    $$('canvas, .textLayer', d).forEach((x) => x.remove());
    d.prepend(canvas);
    const tl = el('div', { class: 'textLayer' }); tl.style.setProperty('--scale-factor', String(S.scale)); d.append(tl);
    const tc = await getText(n);
    try { await pdfjsLib.renderTextLayer({ textContentSource: tc, container: tl, viewport: vp, textDivs: [] }).promise; } catch (e) { /* older API */ }
    highlightMatches(n);
  }
  function unrenderPage(n, force = false) {
    const d = $(`.pdfpage[data-page="${n}"]`); if (!d) return;
    if (d._task) { try { d._task.cancel(); } catch {} d._task = null; }
    if (force || Math.abs(n - S.page) > 6) { $$('canvas, .textLayer', d).forEach((x) => x.remove()); S.rendered.delete(n); }
  }
  async function getText(n) {
    if (!S.textCache.has(n)) { const p = await S.pdf.getPage(n); S.textCache.set(n, await p.getTextContent()); }
    return S.textCache.get(n);
  }
  let activeTimer = null;
  function updateActive() {
    clearTimeout(activeTimer);
    activeTimer = setTimeout(() => {
      if (S.lockUntil && Date.now() < S.lockUntil) return;
      let best = S.page, bestR = -1;
      const vr = viewer.getBoundingClientRect(); const mid = vr.top + vr.height * 0.4;
      $$('.pdfpage').forEach((d) => { const r = d.getBoundingClientRect(); if (r.bottom < vr.top || r.top > vr.bottom) return; const score = -Math.abs((r.top + r.bottom) / 2 - mid) + (r.top <= mid && r.bottom >= mid ? 1e6 : 0); if (score > bestR) { bestR = score; best = +d.dataset.page; } });
      if (best !== S.page) setPage(best, { fromScroll: true });
    }, 80);
  }
  function scrollToPage(n, smooth = true) {
    const d = $(`.pdfpage[data-page="${n}"]`); if (!d) return;
    S.lockUntil = Date.now() + (smooth ? 900 : 400);        // ignore active-page detection while we scroll
    viewer.scrollTo({ top: d.offsetTop - 12, behavior: smooth ? 'smooth' : 'auto' });
    clearTimeout(S.lockTimer); S.lockTimer = setTimeout(() => { S.lockUntil = 0; updateActive(); }, smooth ? 950 : 450);
  }
  function setPage(n, { fromScroll = false, scroll = !fromScroll } = {}) {
    n = Math.max(1, Math.min(S.pageCount || 650, n | 0));
    S.page = n; store.set('page', n);
    $('#page-input').value = n; $('#printed-label').textContent = S.meta.page_map[n]?.printed ? `(printed ${S.meta.page_map[n].printed})` : '';
    $$('.pdfpage.active').forEach((d) => d.classList.remove('active')); $(`.pdfpage[data-page="${n}"]`)?.classList.add('active');
    $$('#thumbs .thumb.active').forEach((d) => d.classList.remove('active')); $(`#thumbs .thumb[data-page="${n}"]`)?.classList.add('active');
    const ch = S.meta.page_map[n]?.chapter; const sel = $('#chapter-select'); if (ch && sel.value !== ch) sel.value = ch;
    history.replaceState(null, '', `#p=${n}`);
    if (scroll) scrollToPage(n);
    if (S.sync || !fromScroll) loadNotes(n);
  }
  function gotoPage(n) { setPage(n, { scroll: true }); }

  // Book search (text layer)
  async function pdfSearch(q) {
    S.matches = []; S.matchIdx = -1; $$('.textLayer .hl').forEach((x) => x.classList.remove('hl', 'cur'));
    const st = $('#pdf-search-status'); if (!q) { st.textContent = ''; return; }
    st.textContent = '…'; const ql = q.toLowerCase();
    for (let n = 1; n <= S.pageCount; n++) {
      const tc = await getText(n); const text = tc.items.map((i) => i.str).join(' ');
      let idx = -1; let count = 0; while ((idx = text.toLowerCase().indexOf(ql, idx + 1)) >= 0 && count < 50) { S.matches.push({ page: n, idx }); count++; }
      if (n % 50 === 0) st.textContent = `${S.matches.length} (${n}/${S.pageCount})`;
    }
    st.textContent = S.matches.length ? `${S.matches.length} hits` : 'no hits';
    $$('.pdfpage').forEach((d) => highlightMatches(+d.dataset.page));
    if (S.matches.length) nextMatch(1);
  }
  function nextMatch(dir) {
    if (!S.matches.length) return;
    S.matchIdx = (S.matchIdx + dir + S.matches.length) % S.matches.length;
    const m = S.matches[S.matchIdx]; $('#pdf-search-status').textContent = `${S.matchIdx + 1}/${S.matches.length}`;
    setPage(m.page, { scroll: true });
    setTimeout(() => { $$('.textLayer .hl.cur').forEach((x) => x.classList.remove('cur')); const d = $(`.pdfpage[data-page="${m.page}"]`); const hs = $$('.textLayer .hl', d); const k = S.matches.filter((x) => x.page === m.page).indexOf(m); hs[k]?.classList.add('cur'); hs[k]?.scrollIntoView({ block: 'center', behavior: 'smooth' }); }, 400);
  }
  function highlightMatches(n) {
    const q = $('#pdf-search').value.trim().toLowerCase(); if (!q) return;
    const d = $(`.pdfpage[data-page="${n}"]`); const spans = $$('.textLayer span', d); if (!spans.length) return;
    spans.forEach((s) => { if (s.textContent.toLowerCase().includes(q)) s.classList.add('hl'); });
  }
  async function locateInBook(excerpt) {
    const q = excerpt.replace(/[“”…]/g, '').trim().split(/\s+/).slice(0, 6).join(' ').toLowerCase();
    setPage(S.pkgPage, { scroll: true });
    await renderPage(S.pkgPage);
    const d = $(`.pdfpage[data-page="${S.pkgPage}"]`); const spans = $$('.textLayer span', d);
    $$('.textLayer .hl', d).forEach((x) => x.classList.remove('hl', 'cur'));
    const first = q.split(' ').slice(0, 3).join(' ');
    const hit = spans.find((s) => s.textContent.toLowerCase().includes(first)) || spans.find((s) => s.textContent.toLowerCase().includes(q.split(' ')[0]));
    if (hit) { hit.classList.add('hl', 'cur'); hit.scrollIntoView({ block: 'center', behavior: 'smooth' }); setTimeout(() => hit.classList.remove('hl', 'cur'), 4000); } else toast('Passage not located on the page text layer');
  }
  // thumbnails
  async function toggleThumbs() {
    S.thumbsOn = !S.thumbsOn; const t = $('#thumbs'); t.hidden = !S.thumbsOn; if (!S.thumbsOn) return;
    if (!t.childElementCount) {
      for (let i = 1; i <= S.pageCount; i++) t.append(el('div', { class: 'thumb', 'data-page': i, onclick: () => gotoPage(i) }, el('canvas', { width: 120, height: Math.round(120 * S.base.h / S.base.w) }), el('div', {}, `${i}${S.meta.page_map[i]?.printed ? ' · ' + S.meta.page_map[i].printed : ''}`)));
      const tio = new IntersectionObserver((ents) => ents.forEach(async (e) => { if (!e.isIntersecting) return; const d = e.target; if (d._done) return; d._done = true; const p = await S.pdf.getPage(+d.dataset.page); const vp = p.getViewport({ scale: 120 / S.base.w }); const c = $('canvas', d); c.width = vp.width; c.height = vp.height; await p.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise; }), { root: t, rootMargin: '400px' });
      $$('.thumb', t).forEach((d) => tio.observe(d));
    }
    $(`#thumbs .thumb[data-page="${S.page}"]`)?.classList.add('active');
    $(`#thumbs .thumb[data-page="${S.page}"]`)?.scrollIntoView({ block: 'center' });
  }

  // ---------------------------------------------------------------- modal, search, graph, quiz, chapters
  function openModal(content) { $('#modal-body').replaceChildren(...[content].flat()); $('#modal').hidden = false; }
  function closeModal() { $('#modal').hidden = true; $('#modal-body').replaceChildren(); }
  async function ensureSearch() {
    if (S.search) return;
    const docs = await fetchJSON('data/search/docs.json');
    S.searchDocs = docs;
    S.search = new MiniSearch({ fields: ['title', 'text'], storeFields: ['title', 'kind', 'page', 'chapter', 'section', 'snippet', 'anchor'], searchOptions: { boost: { title: 2 }, prefix: true, fuzzy: 0.15, combineWith: 'AND' } });
    S.search.addAll(docs);
  }
  async function openSearch(q) {
    const body = el('div'); openModal(body);
    body.append(el('h2', {}, 'Search'), el('p', { class: 'meta' }, 'Loading index…'));
    await ensureSearch();
    const input = el('input', { type: 'search', value: q || '', placeholder: 'concepts, formulas, code, chapters, page numbers…', style: 'width:100%;font-size:16px' });
    const kinds = ['page', 'concept', 'math', 'code', 'question', 'book', 'chapter'];
    const kindSel = el('select', {}, el('option', { value: '' }, 'all types'), ...kinds.map((k) => el('option', { value: k }, k)));
    const chSel = el('select', {}, el('option', { value: '' }, 'all chapters'), ...S.meta.chapters.map((c) => el('option', { value: c.title }, c.title)));
    const res = el('div', { class: 'results' });
    body.replaceChildren(el('h2', {}, 'Search the book and the notes'), input, el('div', { class: 'filters' }, kindSel, chSel, el('span', { class: 'meta' }, 'Tip: type a page number to jump, e.g. "p. 61"')), res);
    const run = () => {
      const qq = input.value.trim(); res.replaceChildren(); if (!qq) return;
      const pm = qq.match(/^(?:p\.?\s*|page\s*)?(\d{1,3})$/i); if (pm) res.append(el('div', { class: 'r', onclick: () => { closeModal(); gotoPage(+pm[1]); } }, el('span', { class: 'kind' }, 'page'), `Go to PDF page ${pm[1]}`));
      let hits = S.search.search(qq, { filter: (d) => (!kindSel.value || d.kind === kindSel.value) && (!chSel.value || d.chapter === chSel.value) });
      if (!hits.length) hits = S.search.search(qq, { combineWith: 'OR', filter: (d) => (!kindSel.value || d.kind === kindSel.value) && (!chSel.value || d.chapter === chSel.value) });
      hits.slice(0, 80).forEach((h) => res.append(el('div', { class: 'r', onclick: () => { closeModal(); gotoPage(h.page); if (h.anchor) setTimeout(() => { const t = $(`#${h.anchor}`); if (t) { t.closest('details')?.setAttribute('open', ''); t.scrollIntoView({ block: 'start', behavior: 'smooth' }); } }, 700); } },
        el('div', {}, el('span', { class: 'kind' }, h.kind), el('b', {}, h.title)), el('div', { class: 'meta' }, `${h.chapter}${h.section ? ' › ' + h.section : ''} · PDF p.${h.page}${S.meta.page_map[h.page]?.printed ? ' (printed ' + S.meta.page_map[h.page].printed + ')' : ''}`), el('div', {}, h.snippet))));
      if (!hits.length && !pm) res.append(el('p', { class: 'meta' }, 'No results.'));
    };
    input.addEventListener('input', run); kindSel.addEventListener('change', run); chSel.addEventListener('change', run); run(); input.focus();
  }
  async function openGraph(focusId) {
    const g = await fetchJSON('data/knowledge_graph/graph.json');
    const wrap = el('div', { id: 'kg' }); const side = el('div', { id: 'kg-side' }); const svgEl = d3.create('svg').attr('viewBox', [0, 0, 760, 640]); wrap.append(svgEl.node(), side);
    openModal([el('h2', {}, `Knowledge graph `, el('span', { class: 'meta' }, `${g.nodes.length} concepts · ${g.links.length} links · click a node`)), wrap]);
    const nodes = g.nodes.map((d) => ({ ...d })), links = g.links.map((d) => ({ ...d }));
    const color = d3.scaleOrdinal(d3.schemeTableau10).domain([...new Set(nodes.map((n) => n.chapter_number))]);
    const sim = d3.forceSimulation(nodes).force('link', d3.forceLink(links).id((d) => d.id).distance(50).strength(.4)).force('charge', d3.forceManyBody().strength(-120)).force('center', d3.forceCenter(380, 320)).force('collide', d3.forceCollide(14));
    const gRoot = svgEl.append('g');
    svgEl.call(d3.zoom().scaleExtent([.3, 4]).on('zoom', (e) => gRoot.attr('transform', e.transform)));
    const link = gRoot.append('g').selectAll('line').data(links).join('line').attr('class', 'link');
    const node = gRoot.append('g').selectAll('g').data(nodes).join('g').attr('class', 'node').call(d3.drag().on('start', (e, d) => { if (!e.active) sim.alphaTarget(.3).restart(); d.fx = d.x; d.fy = d.y; }).on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; }).on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = d.fy = null; }));
    node.append('circle').attr('r', (d) => 4 + Math.min(10, Math.sqrt(d.pages.length) * 2)).attr('fill', (d) => color(d.chapter_number)).on('click', (e, d) => show(d));
    node.append('text').attr('dx', 8).attr('dy', 3).text((d) => d.name);
    sim.on('tick', () => { link.attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y).attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y); node.attr('transform', (d) => `translate(${d.x},${d.y})`); });
    function show(d) {
      node.select('circle').attr('stroke', (n) => n.id === d.id ? 'var(--fg)' : 'var(--panel)').attr('stroke-width', (n) => n.id === d.id ? 3 : 1.5);
      link.attr('stroke-opacity', (l) => l.source.id === d.id || l.target.id === d.id ? .9 : .15);
      const neigh = links.filter((l) => l.source.id === d.id || l.target.id === d.id).map((l) => l.source.id === d.id ? l.target : l.source);
      side.replaceChildren(el('h3', {}, d.name), el('div', { class: 'md', html: md(d.definition) }), el('h5', {}, 'Pages'), el('div', {}, ...d.pages.slice(0, 40).map((p) => [el('a', { class: 'pageref', 'data-page': p, onclick: () => closeModal() }, `p.${p}`), ' '])), el('h5', {}, 'Connected concepts'), el('ul', {}, ...neigh.map((n) => el('li', {}, el('a', { href: '#', onclick: (e) => { e.preventDefault(); show(n); } }, n.name)))));
      renderMath(side);
    }
    const f = focusId && nodes.find((n) => n.id === focusId); if (f) setTimeout(() => show(f), 50); else side.append(el('p', { class: 'meta' }, 'Click a concept to see its definition, the pages where it appears, and the concepts it connects to. Colours = chapter where the concept first appears. Drag nodes, scroll to zoom.'));
  }
  async function openQuiz() {
    const idx = await fetchJSON('data/quizzes/index.json');
    const body = el('div'); openModal(body);
    const chSel = el('select', {}, ...idx.chapters.map((c) => el('option', { value: c.file }, `${c.title} (${c.count} questions)`)));
    const nSel = el('select', {}, ...[5, 10, 20, 40].map((n) => el('option', { value: n }, `${n} questions`)));
    const curOnly = el('label', {}, el('input', { type: 'checkbox' }), ' current page only');
    const area = el('div'); const scoreEl = el('div', { class: 'score' });
    body.append(el('h2', {}, 'Quiz'), el('div', { class: 'quiz-opts' }, chSel, nSel, curOnly, el('button', { class: 'primary', onclick: start }, 'Start')), scoreEl, area);
    async function start() {
      area.replaceChildren(); scoreEl.textContent = '';
      let qs;
      if ($('input', curOnly).checked) { qs = (S.pkg?.questions || []).map((q) => ({ ...q, page: S.pkgPage })); if (!qs.length) { area.append(el('p', {}, 'No questions for the current page.')); return; } }
      else { const data = await fetchJSON('data/quizzes/' + chSel.value); qs = data.questions.slice(); for (let i = qs.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [qs[i], qs[j]] = [qs[j], qs[i]]; } qs = qs.slice(0, +nSel.value); }
      let right = 0, answered = 0;
      qs.forEach((q, i) => { const c = questionCard({ ...q, page_refs: q.page_refs?.length ? q.page_refs : [q.page] }, true); c.prepend(el('div', { class: 'meta' }, `Question ${i + 1} of ${qs.length}`)); c.addEventListener('answered', (e) => { answered++; if (e.detail) right++; scoreEl.textContent = `Score: ${right}/${answered} multiple-choice answered correctly`; }); area.append(c); });
    }
    const curCh = S.meta.page_map[S.page]?.chapter; const cur = idx.chapters.find((c) => c.title === curCh);
    const pick = (cur && cur.count) ? cur : idx.chapters.find((c) => c.count > 0); if (pick) chSel.value = pick.file;
  }
  async function openChapters() {
    const idx = await fetchJSON('data/chapters/index.json');
    const body = el('div', {}, el('h2', {}, 'Chapters'));
    idx.chapters.forEach((c) => body.append(el('div', { class: 'chapter-card' }, el('b', {}, el('a', { class: 'pageref', 'data-page': c.start, onclick: () => closeModal() }, c.title)), el('div', { class: 'meta' }, `PDF pages ${c.start}–${c.end} · ${c.generated}/${c.pages} pages with notes · ${c.concepts} concepts · ${c.questions} questions`), el('div', { class: 'md', html: md(c.overview || '') }), el('div', { class: 'bar' }, el('i', { style: `width:${Math.round(100 * c.generated / c.pages)}%` })))));
    openModal(body);
  }
  function openHelp() {
    openModal(el('div', {}, el('h2', {}, 'How to use this companion'), el('div', { class: 'md', html: md(`
**Left panel** — AI-generated study notes for the page shown on the right. Sections are collapsible; use the chips at the top to jump, and the "Find in these notes" box to filter. Tags mark provenance: **book** = directly supported by the book, **AI explanation** / **analogy** / **inference** = added by the AI during preparation and not attributable to the authors.

**Right panel** — the original book (PDF.js). Scroll, zoom, fit width/page, thumbnails, find in book, full screen. The two panels stay in sync unless you untick *sync panels*.

**Keyboard**: \`←\` / \`→\` previous/next page · \`+\` / \`-\` zoom · \`f\` full screen · \`/\` search · \`g\` knowledge graph · \`q\` quiz · \`t\` thumbnails · \`Esc\` close.

**Offline**: everything (PDF renderer, math, diagrams, search index, all notes) is bundled locally. No AI model, no network. Page numbers in notes are **PDF page numbers**; printed page numbers are shown alongside.
`) })));
  }

  // ---------------------------------------------------------------- wiring
  async function boot() {
    document.documentElement.dataset.theme = store.get('theme', 'auto');
    S.meta = await fetchJSON('data/book_metadata.json');
    const sel = $('#chapter-select');
    S.meta.chapters.forEach((c) => sel.append(el('option', { value: c.title }, c.title)));
    sel.addEventListener('change', () => { const c = S.meta.chapters.find((x) => x.title === sel.value); if (c) gotoPage(c.start); });
    S.sync = store.get('sync', true); $('#sync-toggle').checked = S.sync;
    $('#sync-toggle').addEventListener('change', (e) => { S.sync = e.target.checked; store.set('sync', S.sync); if (S.sync) loadNotes(S.page); });
    $('#page-input').addEventListener('change', (e) => gotoPage(+e.target.value));
    $('#btn-prev').addEventListener('click', () => gotoPage(S.page - 1)); $('#btn-next').addEventListener('click', () => gotoPage(S.page + 1));
    $('#btn-zoom-in').addEventListener('click', () => { S.fit = 'custom'; S.scale = Math.min(4, S.scale * 1.15); applyZoom(); });
    $('#btn-zoom-out').addEventListener('click', () => { S.fit = 'custom'; S.scale = Math.max(.3, S.scale / 1.15); applyZoom(); });
    $('#btn-fit-width').addEventListener('click', () => { S.fit = 'width'; applyZoom(); }); $('#btn-fit-page').addEventListener('click', () => { S.fit = 'page'; applyZoom(); });
    $('#btn-thumbs').addEventListener('click', toggleThumbs);
    $('#btn-fullscreen').addEventListener('click', () => { if (document.fullscreenElement) document.exitFullscreen(); else $('#right').requestFullscreen(); });
    document.addEventListener('fullscreenchange', () => setTimeout(() => applyZoom(), 100));
    let st; $('#pdf-search').addEventListener('input', (e) => { clearTimeout(st); st = setTimeout(() => pdfSearch(e.target.value.trim()), 500); });
    $('#pdf-search').addEventListener('keydown', (e) => { if (e.key === 'Enter') nextMatch(e.shiftKey ? -1 : 1); });
    $('#btn-pdf-prev').addEventListener('click', () => nextMatch(-1)); $('#btn-pdf-next').addEventListener('click', () => nextMatch(1));
    $('#note-search').addEventListener('input', applyNoteSearch);
    $('#global-search').addEventListener('keydown', (e) => { if (e.key === 'Enter') { openSearch(e.target.value); e.target.value = ''; } });
    $('#global-search').addEventListener('focus', (e) => { openSearch(e.target.value); e.target.blur(); });
    $('#btn-graph').addEventListener('click', () => openGraph()); $('#btn-quiz').addEventListener('click', openQuiz); $('#btn-chapters').addEventListener('click', openChapters); $('#btn-help').addEventListener('click', openHelp);
    $('#btn-theme').addEventListener('click', () => { const cur = document.documentElement.dataset.theme; const next = cur === 'dark' ? 'light' : cur === 'light' ? 'auto' : 'dark'; document.documentElement.dataset.theme = next; store.set('theme', next); toast('Theme: ' + next); mermaidReady = false; });
    $('#modal-close').addEventListener('click', closeModal); $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
    document.addEventListener('click', (e) => { const a = e.target.closest('a.pageref[data-page]'); if (a) { e.preventDefault(); if (a.closest('#modal')) closeModal(); gotoPage(+a.dataset.page); } });
    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea, select') || e.metaKey || e.ctrlKey) { if (e.key === 'Escape') { e.target.blur(); closeModal(); } return; }
      const k = e.key;
      if (k === 'ArrowLeft') gotoPage(S.page - 1); else if (k === 'ArrowRight') gotoPage(S.page + 1);
      else if (k === '+' || k === '=') $('#btn-zoom-in').click(); else if (k === '-') $('#btn-zoom-out').click();
      else if (k === 'f') $('#btn-fullscreen').click(); else if (k === '/') { e.preventDefault(); openSearch(''); }
      else if (k === 'g') openGraph(); else if (k === 'q') openQuiz(); else if (k === 't') toggleThumbs(); else if (k === 'Escape') closeModal(); else return;
    });
    // resizable divider
    const div = $('#divider'), left = $('#left'); const savedW = store.get('leftw'); if (savedW) left.style.flexBasis = savedW;
    div.addEventListener('pointerdown', (e) => { div.setPointerCapture(e.pointerId); const move = (ev) => { const r = $('#workspace').getBoundingClientRect(); const w = Math.max(280, Math.min(r.width - 320, ev.clientX - r.left)); left.style.flexBasis = w + 'px'; }; const up = () => { div.removeEventListener('pointermove', move); div.removeEventListener('pointerup', up); store.set('leftw', left.style.flexBasis); applyZoom(); }; div.addEventListener('pointermove', move); div.addEventListener('pointerup', up); });
    new ResizeObserver(() => { if (S.fit !== 'custom' && S.base) applyZoom(); }).observe(viewer);
    await initPDF();
    const hash = location.hash.match(/p=(\d+)/); const start = hash ? +hash[1] : store.get('page', 1);
    setPage(start, { scroll: false }); scrollToPage(start, false);
    viewer.addEventListener('wheel', () => { S.userScrolled = true; }, { passive: true }); viewer.addEventListener('touchstart', () => { S.userScrolled = true; }, { passive: true });
    [400, 1000, 2000].forEach((t) => setTimeout(() => { if (!S.userScrolled && S.page === start) scrollToPage(start, false); }, t));
    window.addEventListener('hashchange', () => { const m = location.hash.match(/p=(\d+)/); if (m && +m[1] !== S.page) gotoPage(+m[1]); });
  }
  boot().catch((e) => { $('#notes').innerHTML = `<div class="placeholder">Failed to start: ${esc(e.message)}.<br>Open this app through the local server (<code>python3 serve.py</code>) — browsers block <code>file://</code> data loading.</div>`; console.error(e); });
  window.BC = { gotoPage, openGraph, openSearch, S };
})();
