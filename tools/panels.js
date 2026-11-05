/* Shared logic for the FRAME museum image templates.
   Reads window.MUSEUM (../data/papers.js) at render time, waits for DM Sans,
   fits the text, then reports on <body>:
     data-ready="1", data-fit="ok" | "overflow" | "error", data-font-px,
   plus diagnostics (data-title-px, data-fonts, data-slack, data-error). */
(function () {
  'use strict';

  var M = window.MUSEUM;
  var Q = new URLSearchParams(location.search);
  var body = document.body;
  var SVGNS = 'http://www.w3.org/2000/svg';

  /* ---------- icons (DM Sans has no arrow glyphs, so they are drawn) ---------- */

  var ICONS = {
    // the "open / read" arrow used in the round Read badge
    arrow: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">'
      + '<path d="M7.5 16.5 16.5 7.5M9.5 7.5h7v7" fill="none" stroke="currentColor" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    // a pointer arrowhead with click rays
    pointer: '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">'
      + '<path d="M9.6 9.6 14.4 21.4 16.5 16.5 21.4 14.4Z" fill="currentColor" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>'
      + '<path d="M9.6 6.2V3.2M6.2 9.6H3.2M12 7.2 14.1 5.1M7.2 12 5.1 14.1" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
  };

  function fillIcons() {
    document.querySelectorAll('i[data-icon]').forEach(function (slotEl) {
      var t = document.createElement('template');
      t.innerHTML = ICONS[slotEl.dataset.icon] || '';
      slotEl.replaceWith(t.content);
    });
  }

  /* ---------- helpers ---------- */

  function slot(name) {
    var el = document.querySelector('[data-slot="' + name + '"]');
    if (!el) throw new Error('template is missing data-slot="' + name + '"');
    return el;
  }
  function setText(name, value) {
    var el = slot(name);
    el.textContent = value;
    return el;
  }

  /* Keep hyphenated compounds ("gender-incongruent") and number ranges
     ("326\u2013329") on one line. U+2060 WORD JOINER is invisible and forbids a
     break; U+2011 is not used because DM Sans has no glyph for it. */
  function keepJoined(s) {
    return String(s)
      .replace(/(\S)-(?=\S)/g, '$1-\u2060')
      .replace(/(\d)\u2013(?=\d)/g, '$1\u2013\u2060');
  }

  /* Typographic quotes and dashes. The words stay exactly as in papers.js. */
  function typo(s) {
    return keepJoined(String(s == null ? '' : s)
      .replace(/(^|[\s([{\u2013\u2014-])"/g, '$1\u201C')
      .replace(/"/g, '\u201D')
      .replace(/(^|[\s([{\u201C\u2013\u2014-])'/g, '$1\u2018')
      .replace(/'/g, '\u2019')
      .replace(/ - /g, ' \u2013 '));
  }

  /* Bind the last two words so a paragraph never ends on a lone word. */
  function noRunt(s) {
    return String(s).replace(/ (\S+)\s*$/, '\u00A0$1');
  }

  function theme(id) {
    var t = (M.themes || []).filter(function (x) { return x.id === id; })[0];
    if (!t) throw new Error('unknown theme id: ' + id);
    return t;
  }
  function paper() {
    var id = Q.get('p') || (M.papers[0] && M.papers[0].id);
    var p = M.papers.filter(function (x) { return x.id === id; })[0];
    if (!p) throw new Error('unknown paper id: ' + id);
    return p;
  }
  function px(n) { return n + 'px'; }
  function setVar(name, value) { body.style.setProperty(name, value); }

  /* Venue and year, without repeating the year when the venue already names it. */
  function venueYear(p) {
    var y = p.year == null ? '' : String(p.year);
    var v = p.venue || '';
    if (!y) return v;
    if (!v) return y;
    if (new RegExp('(^|\\D)' + y + '(\\D|$)').test(v)) return v;
    return v + ' \u00B7 ' + y;
  }

  /* The footer strip bleeds to the card's bottom edge; nothing may pass it. */
  function cardBottom(card) {
    return card.getBoundingClientRect().bottom;
  }
  function lineCount(el, fontPx, lineHeight) {
    return Math.round(el.getBoundingClientRect().height / (fontPx * lineHeight));
  }

  /* ---------- fonts ---------- */

  function fontsReady() {
    void body.offsetHeight; // lay out once so the faces the text needs start loading
    var sample = body.textContent;
    return Promise.all([400, 500, 600, 700].map(function (w) {
      return document.fonts.load(w + ' 40px "DM Sans"', sample).catch(function () { return []; });
    })).then(function () {
      return document.fonts.ready;
    }).then(function () {
      return Array.from(document.fonts).some(function (f) {
        return f.family.replace(/["']/g, '') === 'DM Sans' && f.status === 'loaded';
      });
    });
  }

  function finish(result) {
    var d = body.dataset;
    var extra = result.extra || {};
    Object.keys(extra).forEach(function (k) { d[k] = String(extra[k]); });
    d.fit = result.fit;
    d.fontPx = String(result.px);
    d.ready = '1';
  }
  function fail(e) {
    body.dataset.error = String((e && e.message) || e);
    finish({ fit: 'error', px: 0 });
  }

  function run(build, fit) {
    try {
      if (!M || !M.papers || !M.themes) throw new Error('window.MUSEUM not found (../data/papers.js)');
      build();
      fillIcons();
    } catch (e) { fail(e); return; }
    fontsReady().then(function (ok) {
      body.dataset.fonts = ok ? 'dm-sans' : 'fallback';
      finish(fit());
    }).catch(fail);
  }

  /* ---------- A. abstract panel ---------- */

  var LABEL = /^([A-Z][A-Za-z]*(?:[ -][A-Za-z]+){0,3}):\s+/;

  function buildAbstract(el, paras) {
    paras = Array.isArray(paras) ? paras : [paras];
    var structured = paras.filter(function (s) { return LABEL.test(s); }).length >= 2;
    paras.forEach(function (s) {
      var p = document.createElement('p');
      var m = structured ? String(s).match(LABEL) : null;
      if (m) {
        var b = document.createElement('b');
        b.textContent = m[1] + ':';
        p.appendChild(b);
        p.appendChild(document.createTextNode(' ' + noRunt(typo(String(s).slice(m[0].length)))));
      } else {
        p.textContent = noRunt(typo(s));
      }
      el.appendChild(p);
    });
  }

  /* "doi.org/<doi> · <license>". The DOI never wraps; the line may only
     break after the separator or inside the license wording. */
  function setMeta(p) {
    var el = slot('meta');
    var parts = [];
    el.textContent = '';
    if (p.doi) {
      var doi = document.createElement('span');
      doi.className = 'doi';
      doi.textContent = 'doi.org/' + p.doi;
      parts.push(doi);
    }
    if (p.license && p.license.name) parts.push(document.createTextNode(keepJoined(p.license.name)));
    parts.forEach(function (node, i) {
      if (i) el.appendChild(document.createTextNode(' · '));
      el.appendChild(node);
    });
  }

  var PANEL = { titleMax: 62, titleMin: 52, titleLines: 3, absMax: 36, absMin: 26 };

  /* One title size for the whole series: the largest size that keeps every
     title to three lines. A title that needs more than three lines even at
     the minimum is left out, so it cannot shrink the others; it gets the minimum. */
  function seriesTitleSize(titleEl, ownTitle) {
    var probe = titleEl.cloneNode(false);
    probe.removeAttribute('data-slot');
    probe.style.cssText = 'position:absolute;visibility:hidden;left:0;top:0;margin:0;width:' + titleEl.clientWidth + 'px';
    titleEl.parentNode.appendChild(probe);
    function largest(s) {
      probe.textContent = s;
      for (var t = PANEL.titleMax; t >= PANEL.titleMin; t--) {
        probe.style.fontSize = px(t);
        if (lineCount(probe, t, 1.1) <= PANEL.titleLines) return t;
      }
      return null;
    }
    var own = largest(ownTitle);
    var sizes = M.papers.map(function (p) { return largest(typo(p.title)); })
      .filter(function (t) { return t !== null; });
    probe.remove();
    if (own === null || !sizes.length) return PANEL.titleMin;
    return Math.min.apply(null, sizes);
  }

  function panel() {
    run(function () {
      var p = paper();
      body.dataset.theme = p.theme;
      setText('theme', theme(p.theme).name);
      setText('title', typo(p.title));
      setText('authors', keepJoined((p.authors || []).join(', ')));
      setText('venue', typo(venueYear(p)));
      buildAbstract(slot('abstract'), p.abstract || []);
      setMeta(p);
    }, function () {
      var card = document.querySelector('.card');
      var title = slot('title');
      var abs = slot('abstract');
      var foot = document.querySelector('.foot');
      function textHeight() {
        return abs.lastElementChild.getBoundingClientRect().bottom - abs.getBoundingClientRect().top;
      }
      function slack() { return Math.floor(abs.clientHeight - textHeight()); }
      function fits() {
        return abs.clientHeight > 0 && textHeight() <= abs.clientHeight + 0.5 &&
          foot.getBoundingClientRect().bottom <= cardBottom(card) + 0.5;
      }
      var t, a;
      // 1. one title size for the whole series of panels
      var titlePx = seriesTitleSize(title, title.textContent);
      // 2. the largest abstract size that fits; shrink this title only as a last resort
      for (t = titlePx; t >= PANEL.titleMin; t--) {
        setVar('--title-px', px(t));
        for (a = PANEL.absMax; a >= PANEL.absMin; a--) {
          setVar('--abs-px', px(a));
          if (fits()) {
            return { fit: 'ok', px: a, extra: { titlePx: t, titleLines: lineCount(title, t, 1.1), slack: slack() } };
          }
        }
      }
      setVar('--title-px', px(titlePx));
      setVar('--abs-px', px(PANEL.absMin));
      return { fit: 'overflow', px: PANEL.absMin, extra: {
        titlePx: titlePx, titleLines: lineCount(title, titlePx, 1.1), slack: slack() } };
    });
  }

  /* ---------- B. object plaque ---------- */

  var PLAQUE = { titleMax: 64, titleMin: 50, titleLines: 3, summary: [34, 33, 32, 31, 30] };

  function plaque() {
    run(function () {
      var p = paper();
      body.dataset.theme = p.theme;
      setText('theme', theme(p.theme).name);
      setText('title', typo(p.title));
      setText('byline', keepJoined([p.authorsShort, p.year].filter(Boolean).join(' \u00B7 ')));
      setText('summary', noRunt(typo(p.summary)));
    }, function () {
      var card = document.querySelector('.card');
      var title = slot('title');
      var foot = document.querySelector('.foot');
      function fits() { return foot.getBoundingClientRect().bottom <= cardBottom(card) + 0.5; }
      for (var i = 0; i < PLAQUE.summary.length; i++) {
        var s = PLAQUE.summary[i];
        setVar('--summary-px', px(s));
        for (var t = PLAQUE.titleMax; t >= PLAQUE.titleMin; t--) {
          setVar('--title-px', px(t));
          if (lineCount(title, t, 1.1) <= PLAQUE.titleLines && fits()) {
            return { fit: 'ok', px: s, extra: { titlePx: t, titleLines: lineCount(title, t, 1.1) } };
          }
        }
      }
      return { fit: 'overflow', px: PLAQUE.summary[PLAQUE.summary.length - 1], extra: { titlePx: PLAQUE.titleMin } };
    });
  }

  /* ---------- C/D. hall signs and entrance sign ---------- */

  var SIGN = { titleMax: 230, titleMin: 150, titleWidth: 1400, library: 'Research Library' };

  /* Two display lines: break before '&', otherwise at the most balanced space. */
  function signLines(name) {
    var amp = name.indexOf(' & ');
    if (amp > 0) return [name.slice(0, amp), name.slice(amp + 1)];
    var words = name.split(' ');
    if (words.length < 2) return [name];
    var best = 1, bestDiff = Infinity;
    for (var k = 1; k < words.length; k++) {
      var diff = Math.abs(words.slice(0, k).join(' ').length - words.slice(k).join(' ').length);
      if (diff < bestDiff) { bestDiff = diff; best = k; }
    }
    return [words.slice(0, best).join(' '), words.slice(best).join(' ')];
  }

  function setSignTitle(name) {
    var h = slot('sign-title');
    h.textContent = '';
    signLines(name).forEach(function (line, i) {
      if (i) h.appendChild(document.createTextNode(' ')); // keeps textContent readable; not rendered
      var s = document.createElement('span');
      s.textContent = line;
      h.appendChild(s);
    });
    h.setAttribute('aria-label', name);
  }

  /* One title size for all four signs: the widest line of any sign must fit. */
  function sharedSignSize() {
    var names = (M.themes || []).map(function (t) { return t.name; }).concat([SIGN.library]);
    var probe = document.createElement('span');
    probe.className = 'title';
    probe.style.cssText = 'position:absolute;left:-9999px;top:0;white-space:nowrap;font-size:200px';
    document.querySelector('.card').appendChild(probe);
    var widest = 0;
    names.forEach(function (n) {
      signLines(n).forEach(function (line) {
        probe.textContent = line;
        widest = Math.max(widest, probe.getBoundingClientRect().width);
      });
    });
    probe.remove();
    return Math.max(SIGN.titleMin, Math.min(SIGN.titleMax, Math.floor(200 * SIGN.titleWidth / widest)));
  }

  function fitSign() {
    var card = document.querySelector('.card');
    var copy = document.querySelector('.copy');
    var blurb = slot('blurb');
    var size = sharedSignSize();
    function fits() {
      var c = copy.getBoundingClientRect();
      return c.top >= 24 + 60 && c.bottom <= card.getBoundingClientRect().bottom - 60;
    }
    for (var s = size; s >= SIGN.titleMin; s -= 2) {
      setVar('--sign-px', px(s));
      if (fits()) {
        return { fit: 'ok', px: parseFloat(getComputedStyle(blurb).fontSize), extra: { titlePx: s } };
      }
    }
    return { fit: 'overflow', px: parseFloat(getComputedStyle(blurb).fontSize), extra: { titlePx: SIGN.titleMin } };
  }

  function svg(name, attrs, parent) {
    var n = document.createElementNS(SVGNS, name);
    Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }

  /* The sign motif is a doorway: nested arches standing on the bottom edge.
     A hall sign shows its own doorway; the entrance sign shows all three. */
  function arch(g, cx, top, w, fill) {
    var r = w / 2;
    svg('path', {
      d: 'M' + (cx - r) + ' 1000V' + (top + r) + 'A' + r + ' ' + r + ' 0 0 1 ' + (cx + r) + ' ' + (top + r) + 'V1000Z',
      style: 'fill:' + fill
    }, g);
  }
  function doorway(g, cx, top, w, accent, steps) {
    steps.forEach(function (s, i) {
      var inset = i * w * 0.13;
      arch(g, cx, top + inset * 1.6, w - inset * 2, 'color-mix(in srgb, ' + accent + ' ' + s + '%, var(--card))');
    });
  }

  function hall() {
    run(function () {
      var t = theme(Q.get('t') || M.themes[0].id);
      body.dataset.theme = t.id;
      setSignTitle(t.name);
      setText('blurb', noRunt(typo(t.blurb)));
      var g = slot('motif');
      g.setAttribute('viewBox', '0 0 900 1000');
      doorway(g, 500, 170, 620, 'var(--accent)', [8, 13, 19, 27]);
    }, fitSign);
  }

  function librarySign() {
    run(function () {
      setSignTitle(SIGN.library);
      // the hall colours, in hall order, for the band, the rule and the motif
      var ids = M.themes.map(function (t) { return t.id; });
      var band = document.querySelector('.tri-band');
      var rule = document.querySelector('.rule');
      ids.forEach(function (id) {
        var seg = document.createElement('i');
        seg.style.background = 'var(--accent-' + id + ', var(--ink-2))';
        band.appendChild(seg);
        rule.appendChild(seg.cloneNode());
      });
      // the instruction: one sentence per line, each led by the mark visitors
      // meet next (the pointer of the plaque cue, then the panel's round ↗ badge)
      var how = slot('blurb');
      var sentences = how.textContent.trim().split(/(?<=\.)\s+/);
      var marks = ['<i data-icon="pointer"></i>', '<span class="badge"><i data-icon="arrow"></i></span>'];
      how.textContent = '';
      sentences.forEach(function (s, i) {
        if (i) how.appendChild(document.createTextNode(' ')); // keeps textContent readable; not rendered
        var step = document.createElement('span');
        step.className = 'step';
        step.innerHTML = marks[i] || marks[0];
        step.appendChild(document.createTextNode(s));
        how.appendChild(step);
      });
      // one doorway per hall, the last one ending where a hall sign's doorway ends
      var g = slot('motif');
      g.setAttribute('viewBox', '0 0 1200 1000');
      var step = 1100 / ids.length;
      var w = Math.min(300, step - 60);
      ids.forEach(function (id, i) {
        var cx = 1110 - w / 2 - step * (ids.length - 1 - i);
        doorway(g, cx, 330, w, 'var(--accent-' + id + ', var(--ink-2))', [10, 16, 24]);
      });
    }, fitSign);
  }

  window.Panels = { panel: panel, plaque: plaque, hall: hall, librarySign: librarySign };
})();
