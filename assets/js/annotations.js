/* Highlights & personal notes — on the book pages (PDF) and on the companion notes. Fully offline.
 *
 * Storage: every change is written to localStorage immediately and, when the app is served by serve.py,
 * mirrored to data/user/annotations.json through PUT /api/annotations (loopback only). Items are merged by
 * id (newest `updated` wins); deletions are kept as tombstones so they survive merges and imports.
 *
 * Item shapes
 *   PDF highlight  {id, target:'pdf',   kind:'highlight', page, color, note, text, rects:[[x,y,w,h], ...]}  (0..1, page-relative)
 *   PDF pin note   {id, target:'pdf',   kind:'pin',       page, color, note, pos:[x,y]}
 *   Notes highlight{id, target:'notes', kind:'highlight', page, color, note, text, prefix, suffix, occ, section}
 */
(() => {
  'use strict';
  const COLORS = { yellow: '#ffd84d', green: '#6fdc84', blue: '#73b3ff', pink: '#ff8cc0', purple: '#c39bff' };
  const LS_KEY = 'bc:annotations';
  const SKIP = '.katex, .viz, svg, script, style, button, input, textarea, select, .anno-ui';
  const A = { items: new Map(), server: false, pinMode: false, saveTimer: null, pendingFocus: null, bar: null, pop: null, sel: null };
  const BC = () => window.BC;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const nowISO = () => new Date().toISOString();
  const live = () => [...A.items.values()].filter((a) => !a.deleted);
  const pageDiv = (n) => $(`.pdfpage[data-page="${n}"]`);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const h = (tag, attrs = {}, ...kids) => BC().el(tag, attrs, ...kids);

  // ------------------------------------------------------------------ persistence
  function mergeIn(list) {
    let n = 0;
    for (const a of list || []) {
      if (!a || typeof a.id !== 'string') continue;
      const cur = A.items.get(a.id);
      if (!cur || String(a.updated || '') > String(cur.updated || '')) { A.items.set(a.id, a); n++; }
    }
    return n;
  }
  const snapshot = () => ({ version: 1, book: 'dlwp3', saved_at: nowISO(), items: [...A.items.values()] });
  function save() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(snapshot())); } catch { /* storage full or blocked */ }
    clearTimeout(A.saveTimer); A.saveTimer = setTimeout(pushToServer, 500);
    updateCount();
  }
  async function pushToServer() {
    if (!A.server) return;
    try {
      const r = await fetch('api/annotations', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot()) });
      if (!r.ok) throw new Error(r.status);
    } catch { BC().toast('Could not write annotations to disk — they are kept in this browser.'); }
  }
  async function load() {
    try { mergeIn(JSON.parse(localStorage.getItem(LS_KEY) || 'null')?.items); } catch { /* ignore */ }
    try {
      const r = await fetch('api/annotations', { cache: 'no-store' });
      if (r.ok) { A.server = true; mergeIn((await r.json()).items); }
    } catch { /* plain static server: browser storage only */ }
    save();   // push anything that only existed locally to disk
  }
  function upsert(a) { a.updated = nowISO(); A.items.set(a.id, a); save(); redraw(a); return a; }
  function remove(a) {
    A.items.set(a.id, { id: a.id, deleted: true, updated: nowISO() }); save(); redraw(a);
  }
  function redraw(a) {
    if (a.target === 'pdf') drawPage(a.page);
    else if (a.target === 'notes' && a.page === BC().S.pkgPage) paintNotes();
  }

  // ------------------------------------------------------------------ PDF pages
  function drawPage(n) {
    const d = pageDiv(n); if (!d) return;
    $$('.annoLayer, .pinLayer', d).forEach((x) => x.remove());
    const items = live().filter((a) => a.target === 'pdf' && a.page === n);
    if (!items.length) return;
    const L = h('div', { class: 'annoLayer' }), P = h('div', { class: 'pinLayer' });
    for (const a of items) {
      if (a.kind === 'highlight') {
        (a.rects || []).forEach(([x, y, w, hh]) => L.append(h('div', { class: 'hl-rect', 'data-anno': a.id, style: `left:${x * 100}%;top:${y * 100}%;width:${w * 100}%;height:${hh * 100}%;background:${COLORS[a.color] || COLORS.yellow}` })));
        if (a.note && a.rects?.length) {
          const [x, y, w] = a.rects[a.rects.length - 1];
          const b = h('div', { class: 'hl-note', 'data-anno': a.id, title: a.note.slice(0, 200), style: `left:${(x + w) * 100}%;top:${y * 100}%;background:${COLORS[a.color] || COLORS.yellow}` }, '✎');
          b.addEventListener('click', (e) => { e.stopPropagation(); openEditor(a, b.getBoundingClientRect()); });
          P.append(b);
        }
      } else if (a.kind === 'pin') {
        const pin = h('div', { class: 'pin' + (a.note ? '' : ' empty'), 'data-anno': a.id, title: a.note ? a.note.slice(0, 200) : 'Empty note', style: `left:${a.pos[0] * 100}%;top:${a.pos[1] * 100}%;background:${COLORS[a.color] || COLORS.yellow}` }, '✎');
        dragPin(pin, a, d);
        P.append(pin);
      }
    }
    d.append(L, P);
  }
  function drawAllPages() { const pages = new Set(live().filter((a) => a.target === 'pdf').map((a) => a.page)); $$('.pdfpage .annoLayer, .pdfpage .pinLayer').forEach((x) => x.remove()); pages.forEach(drawPage); }

  function dragPin(pin, a, d) {
    pin.addEventListener('pointerdown', (e) => {
      e.stopPropagation(); e.preventDefault(); pin.setPointerCapture(e.pointerId);
      const sx = e.clientX, sy = e.clientY; let moved = false;
      const move = (ev) => {
        if (!moved && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 4) return; moved = true;
        const r = d.getBoundingClientRect();
        pin.style.left = clamp((ev.clientX - r.left) / r.width, 0, 1) * 100 + '%'; pin.style.top = clamp((ev.clientY - r.top) / r.height, 0, 1) * 100 + '%';
      };
      const up = (ev) => {
        pin.removeEventListener('pointermove', move); pin.removeEventListener('pointerup', up);
        if (moved) { const r = d.getBoundingClientRect(); a.pos = [clamp((ev.clientX - r.left) / r.width, 0, 1), clamp((ev.clientY - r.top) / r.height, 0, 1)]; upsert(a); }
        else openEditor(a, pin.getBoundingClientRect());
      };
      pin.addEventListener('pointermove', move); pin.addEventListener('pointerup', up);
    });
    pin.addEventListener('click', (e) => e.stopPropagation());
  }

  function pdfSelection(range, d) {
    const pr = d.getBoundingClientRect();
    const raw = [...range.getClientRects()].filter((r) => r.width > 1 && r.height > 1 && r.bottom > pr.top && r.top < pr.bottom && r.right > pr.left && r.left < pr.right)
      .map((r) => { const l = Math.max(r.left, pr.left), t = Math.max(r.top, pr.top), rr = Math.min(r.right, pr.right), b = Math.min(r.bottom, pr.bottom); return [(l - pr.left) / pr.width, (t - pr.top) / pr.height, (rr - l) / pr.width, (b - t) / pr.height]; });
    raw.sort((p, q) => p[1] - q[1] || p[0] - q[0]);
    const lines = [];
    for (const r of raw) {   // merge fragments on the same line into one rectangle
      const L = lines.find((l) => Math.abs((l[1] + l[3] / 2) - (r[1] + r[3] / 2)) < Math.min(l[3], r[3]) * 0.6 && r[0] <= l[0] + l[2] + 0.02 && r[0] + r[2] >= l[0] - 0.02);
      if (L) { const x0 = Math.min(L[0], r[0]), y0 = Math.min(L[1], r[1]), x1 = Math.max(L[0] + L[2], r[0] + r[2]), y1 = Math.max(L[1] + L[3], r[1] + r[3]); L.splice(0, 4, x0, y0, x1 - x0, y1 - y0); }
      else lines.push(r.slice());
    }
    return lines.map((r) => r.map((v) => Math.round(v * 1e5) / 1e5));
  }
  function hitHighlight(n, x, y) {
    return live().find((a) => a.target === 'pdf' && a.kind === 'highlight' && a.page === n && (a.rects || []).some(([rx, ry, rw, rh]) => x >= rx && x <= rx + rw && y >= ry && y <= ry + rh));
  }
  function onPagesClick(e) {
    if (e.target.closest('.pin, .hl-note')) return;
    const d = e.target.closest('.pdfpage'); if (!d) return;
    const r = d.getBoundingClientRect(); const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height; const n = +d.dataset.page;
    if (A.pinMode) {
      const a = upsert({ id: uid(), target: 'pdf', kind: 'pin', page: n, color: A.lastColor || 'yellow', note: '', pos: [x, y], created: nowISO() });
      setPinMode(e.shiftKey);   // shift-click keeps note mode on
      const pin = $(`.pin[data-anno="${a.id}"]`, d); openEditor(a, pin ? pin.getBoundingClientRect() : { left: e.clientX, right: e.clientX, top: e.clientY, bottom: e.clientY }, true);
      return;
    }
    const sel = getSelection(); if (sel && !sel.isCollapsed) return;
    const a = hitHighlight(n, x, y);
    if (a) { const rr = $(`.hl-rect[data-anno="${a.id}"]`, d); openEditor(a, rr ? rr.getBoundingClientRect() : { left: e.clientX, right: e.clientX, top: e.clientY, bottom: e.clientY }); }
  }
  function setPinMode(on) {
    A.pinMode = on; document.body.classList.toggle('pin-mode', on);
    const b = $('#btn-pin'); if (b) b.classList.toggle('on', on);
    if (on) BC().toast('Note mode: click anywhere on a book page to pin a note (Esc to cancel)', 2600);
  }

  // ------------------------------------------------------------------ companion notes (left panel)
  function linearize(root) {
    const nodes = []; let pos = 0;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) => n.parentElement.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    while (w.nextNode()) { const n = w.currentNode; nodes.push({ n, s: pos }); pos += n.data.length; }
    return { nodes, text: nodes.map((x) => x.n.data).join('') };
  }
  function offsetOf(lin, container, off) {
    if (container.nodeType === 3) { const f = lin.nodes.find((x) => x.n === container); if (f) return f.s + off; }
    const r = document.createRange(); r.setStart(container, off); r.collapse(true);
    for (const x of lin.nodes) if (r.comparePoint(x.n, 0) >= 0) return x.s;
    return lin.text.length;
  }
  function notesSelection(range) {
    const box = $('#notes'); const lin = linearize(box);
    let s = offsetOf(lin, range.startContainer, range.startOffset), e = offsetOf(lin, range.endContainer, range.endOffset);
    while (s < e && /\s/.test(lin.text[s])) s++; while (e > s && /\s/.test(lin.text[e - 1])) e--;
    if (e - s < 2) return null;
    const text = lin.text.slice(s, e);
    let occ = 0, i = -1; while ((i = lin.text.indexOf(text, i + 1)) >= 0 && i < s) occ++;
    const startEl = range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement;
    return { text, prefix: lin.text.slice(Math.max(0, s - 40), s), suffix: lin.text.slice(e, e + 40), occ, section: startEl.closest('details.sec')?.id || '' };
  }
  function locate(lin, a) {
    const hits = []; let i = -1; while ((i = lin.text.indexOf(a.text, i + 1)) >= 0) hits.push(i);
    if (!hits.length) return -1;
    const score = (p) => { let sc = 0; const pre = lin.text.slice(Math.max(0, p - 40), p), suf = lin.text.slice(p + a.text.length, p + a.text.length + 40);
      for (let k = 1; k <= Math.min(pre.length, (a.prefix || '').length); k++) { if (pre[pre.length - k] === a.prefix[a.prefix.length - k]) sc++; else break; }
      for (let k = 0; k < Math.min(suf.length, (a.suffix || '').length); k++) { if (suf[k] === a.suffix[k]) sc++; else break; }
      return sc; };
    let best = hits[Math.min(a.occ || 0, hits.length - 1)], bestS = score(best);
    hits.forEach((p) => { const sc = score(p); if (sc > bestS) { best = p; bestS = sc; } });
    return best;
  }
  function wrap(lin, s, e, a) {
    const marks = [];
    for (const x of lin.nodes) {
      const ns = x.s, ne = x.s + x.n.data.length; if (ne <= s || ns >= e) continue;
      let node = x.n; const a0 = Math.max(s, ns) - ns, a1 = Math.min(e, ne) - ns;
      if (!node.data.slice(a0, a1).trim()) continue;
      if (a1 < node.data.length) node.splitText(a1);
      if (a0 > 0) node = node.splitText(a0);
      const m = document.createElement('mark'); m.className = 'anno'; m.dataset.anno = a.id; m.style.setProperty('--c', COLORS[a.color] || COLORS.yellow);
      node.replaceWith(m); m.append(node); marks.push(m);
    }
    if (marks.length && a.note) { marks[marks.length - 1].classList.add('has-note'); marks[marks.length - 1].title = a.note.slice(0, 200); }
    return marks;
  }
  function unpaintNotes(box) { $$('mark.anno', box).forEach((m) => { const p = m.parentNode; m.replaceWith(...m.childNodes); p.normalize(); }); }
  function paintNotes() {
    const box = $('#notes'); if (!box) return; const n = BC().S.pkgPage;
    unpaintNotes(box);
    const missing = [];
    for (const a of live().filter((x) => x.target === 'notes' && x.page === n)) {
      const lin = linearize(box); const s = locate(lin, a);
      if (s < 0) { missing.push(a); continue; }
      wrap(lin, s, s + a.text.length, a);
    }
    if (missing.length) BC().toast(`${missing.length} highlight(s) on this page could not be re-anchored (text changed) — see My notes`, 3500);
    if (A.pendingFocus) { const id = A.pendingFocus; A.pendingFocus = null; setTimeout(() => focusNotesMark(id), 60); }
  }
  function focusNotesMark(id) {
    const m = $(`#notes mark.anno[data-anno="${id}"]`); if (!m) return;
    m.closest('details')?.setAttribute('open', '');
    m.scrollIntoView({ block: 'center', behavior: 'smooth' }); flash($$(`#notes mark.anno[data-anno="${id}"]`));
  }
  function flash(els) { els.forEach((x) => { x.classList.remove('flash'); void x.offsetWidth; x.classList.add('flash'); }); }

  // ------------------------------------------------------------------ selection toolbar
  function buildBar() {
    const bar = h('div', { id: 'anno-bar', class: 'anno-ui', hidden: true });
    Object.entries(COLORS).forEach(([name, c]) => {
      const b = h('button', { class: 'swatch', title: `Highlight (${name})`, 'data-color': name, style: `background:${c}` });
      b.addEventListener('mousedown', (e) => e.preventDefault());
      b.addEventListener('click', () => createFromSelection(name, false)); bar.append(b);
    });
    const nb = h('button', { class: 'add-note', title: 'Highlight and add a note' }, '✎ Note');
    nb.addEventListener('mousedown', (e) => e.preventDefault()); nb.addEventListener('click', () => createFromSelection(A.lastColor || 'yellow', true));
    bar.append(nb); document.body.append(bar); A.bar = bar;
  }
  function captureSelection() {
    const sel = getSelection(); if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    const startEl = range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement;
    if (!startEl || startEl.closest('.anno-ui')) return null;
    const d = startEl.closest('.pdfpage');
    if (d && startEl.closest('.textLayer')) {
      const rects = pdfSelection(range, d); const text = sel.toString().replace(/\s+/g, ' ').trim();
      if (!rects.length || !text) return null;
      return { target: 'pdf', page: +d.dataset.page, rects, text, range };
    }
    if (startEl.closest('#notes') && !startEl.closest(SKIP)) {
      const info = notesSelection(range); if (!info) return null;
      return { target: 'notes', page: BC().S.pkgPage, ...info, range };
    }
    return null;
  }
  function showBar() {
    const c = captureSelection(); A.sel = c;
    if (!c) { A.bar.hidden = true; return; }
    const rs = [...c.range.getClientRects()].filter((r) => r.width > 0); const r = rs[rs.length - 1] || c.range.getBoundingClientRect();
    A.bar.hidden = false; const bw = A.bar.offsetWidth, bh = A.bar.offsetHeight;
    A.bar.style.left = clamp(r.right - bw / 2, 8, innerWidth - bw - 8) + 'px';
    A.bar.style.top = (r.bottom + bh + 12 < innerHeight ? r.bottom + 8 : Math.max(8, r.top - bh - 8)) + 'px';
  }
  function createFromSelection(color, withNote) {
    const c = A.sel; if (!c) return; A.lastColor = color;
    const base = { id: uid(), kind: 'highlight', color, note: '', created: nowISO(), page: c.page, text: c.text, target: c.target };
    const a = c.target === 'pdf' ? { ...base, rects: c.rects } : { ...base, prefix: c.prefix, suffix: c.suffix, occ: c.occ, section: c.section };
    upsert(a); getSelection().removeAllRanges(); A.bar.hidden = true; A.sel = null;
    if (withNote) {
      const t = a.target === 'pdf' ? $(`.pdfpage[data-page="${a.page}"] .hl-rect[data-anno="${a.id}"]`) : $(`#notes mark.anno[data-anno="${a.id}"]`);
      openEditor(a, t ? t.getBoundingClientRect() : { left: innerWidth / 2, right: innerWidth / 2, top: 120, bottom: 120 }, true);
    }
  }

  // ------------------------------------------------------------------ note editor popover
  function closeEditor() {
    if (!A.pop) return; const { a, ta } = A.pop._ctx;
    commitNote(a, ta.value);
    const cur = A.items.get(a.id);
    if (cur && !cur.deleted && cur.kind === 'pin' && !cur.note.trim()) remove(cur);   // don't leave empty pins behind
    A.pop.remove(); A.pop = null;
  }
  function commitNote(a, text) {
    const cur = A.items.get(a.id); if (!cur || cur.deleted) return;
    if ((cur.note || '') !== text) { cur.note = text; upsert(cur); }
  }
  function openEditor(a, anchor, focus = true) {
    if (A.pop) closeEditor();
    a = A.items.get(a.id) || a;
    const ta = h('textarea', { placeholder: 'Write your note… (Markdown and $math$ supported)', rows: 5 }); ta.value = a.note || '';
    let t; ta.addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => commitNote(a, ta.value), 400); });
    ta.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); closeEditor(); } if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) closeEditor(); });
    const sw = h('div', { class: 'swatches' }, ...Object.entries(COLORS).map(([name, c]) => {
      const b = h('button', { class: 'swatch' + (a.color === name ? ' sel' : ''), title: name, style: `background:${c}` });
      b.addEventListener('click', () => { const cur = A.items.get(a.id); cur.color = name; A.lastColor = name; commitNote(cur, ta.value); upsert(cur); $$('.swatch', sw).forEach((x) => x.classList.toggle('sel', x === b)); });
      return b; }));
    const kindLabel = a.kind === 'pin' ? `Note pinned on book p.${a.page}` : a.target === 'pdf' ? `Highlight on book p.${a.page}` : `Highlight in the notes of p.${a.page}`;
    const pop = h('div', { id: 'anno-pop', class: 'anno-ui' },
      h('div', { class: 'pop-head' }, h('b', {}, kindLabel), h('button', { class: 'icon', title: 'Close (Esc)', onclick: closeEditor }, '✕')),
      a.text ? h('blockquote', {}, a.text.length > 220 ? a.text.slice(0, 220) + '…' : a.text) : '',
      ta, h('div', { class: 'pop-foot' }, sw, h('span', { class: 'grow' }),
        h('button', { class: 'danger', onclick: () => { const cur = A.items.get(a.id); A.pop.remove(); A.pop = null; remove(cur); BC().toast('Deleted'); } }, 'Delete'),
        h('button', { class: 'primary', onclick: closeEditor }, 'Done')));
    pop._ctx = { a, ta }; document.body.append(pop); A.pop = pop;
    const pw = pop.offsetWidth, ph = pop.offsetHeight;
    pop.style.left = clamp(anchor.left, 8, innerWidth - pw - 8) + 'px';
    pop.style.top = (anchor.bottom + ph + 10 < innerHeight ? anchor.bottom + 6 : Math.max(8, anchor.top - ph - 6)) + 'px';
    if (focus) setTimeout(() => ta.focus(), 0);
  }

  // ------------------------------------------------------------------ "My notes" panel
  function updateCount() { const b = $('#btn-annos'); if (b) b.textContent = `My notes (${live().length})`; }
  function goTo(a) {
    const bc = BC(); bc.closeModal();
    if (a.target === 'notes') {
      if (bc.S.pkgPage === a.page) focusNotesMark(a.id); else { A.pendingFocus = a.id; bc.gotoPage(a.page); if (!bc.S.sync) bc.loadNotes(a.page); }
    } else {
      bc.gotoPage(a.page);
      setTimeout(() => { const t = $(`.pdfpage[data-page="${a.page}"] [data-anno="${a.id}"]`); if (t) { t.scrollIntoView({ block: 'center', behavior: 'smooth' }); flash($$(`.pdfpage [data-anno="${a.id}"]`)); } }, 950);
    }
  }
  function openPanel() {
    const bc = BC(); const meta = bc.S.meta;
    const q = h('input', { type: 'search', placeholder: 'Filter your highlights and notes…', style: 'flex:1;min-width:200px' });
    const colorSel = h('select', {}, h('option', { value: '' }, 'all colours'), ...Object.keys(COLORS).map((c) => h('option', { value: c }, c)));
    const scopeSel = h('select', {}, h('option', { value: '' }, 'book + notes'), h('option', { value: 'pdf' }, 'book pages only'), h('option', { value: 'notes' }, 'companion notes only'), h('option', { value: 'with-note' }, 'only items with a note'));
    const file = h('input', { type: 'file', accept: 'application/json,.json', hidden: true });
    const list = h('div', { class: 'anno-list' });
    const status = h('div', { class: 'meta' }, A.server ? 'Saved automatically to data/user/annotations.json and to this browser.' : 'Saved in this browser only (start the app with serve.py to also save to data/user/annotations.json). Use Export for a backup.');
    const exportBtn = h('button', { onclick: () => {
      const blob = new Blob([JSON.stringify(snapshot(), null, 1)], { type: 'application/json' });
      const url = URL.createObjectURL(blob); const link = h('a', { href: url, download: `dlwp-annotations-${new Date().toISOString().slice(0, 10)}.json` }); document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
    } }, 'Export');
    const importBtn = h('button', { onclick: () => file.click() }, 'Import');
    file.addEventListener('change', async () => {
      try { const data = JSON.parse(await file.files[0].text()); const n = mergeIn(Array.isArray(data) ? data : data.items); save(); drawAllPages(); paintNotes(); render(); bc.toast(`Imported ${n} item(s)`); }
      catch (e) { bc.toast('Import failed: ' + e.message); }
      file.value = '';
    });
    const render = () => {
      const qq = q.value.trim().toLowerCase();
      const items = live().filter((a) => (!colorSel.value || a.color === colorSel.value) && (!scopeSel.value || (scopeSel.value === 'with-note' ? !!(a.note || '').trim() : a.target === scopeSel.value)) && (!qq || `${a.text || ''} ${a.note || ''}`.toLowerCase().includes(qq)))
        .sort((x, y) => x.page - y.page || String(x.created).localeCompare(String(y.created)));
      list.replaceChildren();
      if (!items.length) { list.append(h('p', { class: 'meta' }, live().length ? 'Nothing matches the filter.' : 'No highlights or notes yet. Select text on a book page or in the notes and pick a colour, or press “✎ Note” in the viewer toolbar (key n) and click anywhere on a page.')); return; }
      let lastPage = null;
      for (const a of items) {
        if (a.page !== lastPage) { lastPage = a.page; const pm = meta.page_map[a.page] || {}; list.append(h('h4', { class: 'anno-page' }, h('a', { class: 'pageref', 'data-page': a.page }, `PDF p.${a.page}${pm.printed ? ' (printed ' + pm.printed + ')' : ''}`), h('span', { class: 'meta' }, ` ${pm.chapter || ''}${pm.section ? ' › ' + pm.section : ''}`))); }
        const label = a.kind === 'pin' ? 'note pinned on the page' : a.target === 'pdf' ? 'highlight · book' : 'highlight · companion notes';
        const card = h('div', { class: 'anno-card', style: `--c:${COLORS[a.color] || COLORS.yellow}` },
          h('div', { class: 'anno-meta' }, h('span', { class: 'dot' }), label, h('span', { class: 'meta' }, ' · ' + new Date(a.updated || a.created).toLocaleString())),
          a.text ? h('blockquote', {}, a.text.length > 400 ? a.text.slice(0, 400) + '…' : a.text) : '',
          a.note ? h('div', { class: 'md', html: bc.md(a.note) }) : '',
          h('div', { class: 'anno-actions' },
            h('button', { onclick: () => goTo(a) }, 'Go to'),
            h('button', { onclick: () => { goTo(a); setTimeout(() => { const t = a.target === 'notes' ? $(`#notes mark.anno[data-anno="${a.id}"]`) : $(`.pdfpage [data-anno="${a.id}"]`); openEditor(a, t ? t.getBoundingClientRect() : { left: innerWidth / 2, right: innerWidth / 2, top: 140, bottom: 140 }); }, a.target === 'notes' ? 900 : 1300); } }, 'Edit note'),
            h('button', { class: 'danger', onclick: () => { remove(a); render(); } }, 'Delete')));
        list.append(card);
      }
      bc.renderMath(list);
    };
    [q, colorSel, scopeSel].forEach((x) => x.addEventListener(x === q ? 'input' : 'change', render));
    bc.openModal([h('h2', {}, 'My highlights & notes ', h('span', { class: 'meta' }, `${live().length} item(s)`)), status,
      h('div', { class: 'filters' }, q, colorSel, scopeSel, exportBtn, importBtn, file), list]);
    render(); q.focus();
  }

  // ------------------------------------------------------------------ search integration
  function search(q) {
    const qq = q.toLowerCase();
    return live().filter((a) => `${a.text || ''} ${a.note || ''}`.toLowerCase().includes(qq)).sort((x, y) => x.page - y.page).slice(0, 40);
  }

  // ------------------------------------------------------------------ public hooks (called by app.js)
  window.Anno = {
    COLORS, goTo, search, openPanel, setPinMode,
    get pinMode() { return A.pinMode; },
    items: () => live(),
    async init() {
      buildBar();
      await load(); drawAllPages(); updateCount();
      $('#pages').addEventListener('click', onPagesClick);
      document.addEventListener('mouseup', (e) => { if (e.target.closest?.('.anno-ui')) return; setTimeout(showBar, 0); });
      document.addEventListener('keyup', (e) => { if (e.shiftKey && e.key.startsWith('Arrow')) setTimeout(showBar, 0); });
      document.addEventListener('selectionchange', () => { const s = getSelection(); if ((!s || s.isCollapsed) && A.bar) A.bar.hidden = true; });
      document.addEventListener('mousedown', (e) => { if (A.pop && !e.target.closest?.('#anno-pop, .pin, .hl-note')) closeEditor(); }, true);
      document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { if (A.pop) closeEditor(); if (A.pinMode) setPinMode(false); if (A.bar) A.bar.hidden = true; } });
      ['wheel', 'touchmove'].forEach((ev) => $('#viewer').addEventListener(ev, () => { if (A.bar && !A.bar.hidden) A.bar.hidden = true; }, { passive: true }));
      $('#notes').addEventListener('click', (e) => { const m = e.target.closest('mark.anno'); if (!m) return; const s = getSelection(); if (s && !s.isCollapsed) return; const a = A.items.get(m.dataset.anno); if (a) openEditor(a, m.getBoundingClientRect()); });
    },
    pageRendered(n) { if (!pageDiv(n)?.querySelector('.annoLayer')) drawPage(n); },
    notesRendered() { paintNotes(); },
  };
})();
