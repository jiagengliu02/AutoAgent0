(() => {
  'use strict';
  const data = window.AUTOAGENT;
  if (!data) return;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  // Relative assets work both on a hosted site and when index.html is opened locally.
  const safeURL = value => {
    if (!value || typeof value !== 'string') return '';
    const trimmed = value.trim();
    if (!trimmed || /[\u0000-\u001f]/.test(trimmed)) return '';
    try {
      const url = new URL(trimmed, document.baseURI);
      const relativeFilePreview = new URL(document.baseURI).protocol === 'file:'
        && url.protocol === 'file:'
        && !/^(?:[a-z][a-z0-9+.-]*:|[\\/])/i.test(trimmed);
      return ['http:', 'https:'].includes(url.protocol) || relativeFilePreview ? trimmed : '';
    } catch { return ''; }
  };
  const icon = '<span class="placeholder-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="3" y="5" width="13" height="14" rx="2"/><path d="m16 10 5-3v10l-5-3"/></svg></span>';

  function placeholder(slot, item, failed = false) {
    slot.replaceChildren();
    const poster = safeURL(item.poster);
    if (poster) {
      const img = document.createElement('img');
      img.src = poster; img.alt = ''; img.className = 'video-poster';
      img.loading = slot.closest('.hero-media') ? 'eager' : 'lazy';
      slot.append(img);
    }
    const shade = document.createElement('div'); shade.className = 'placeholder-shade';
    const content = document.createElement('div'); content.className = 'placeholder-content';
    content.innerHTML = `${icon}<span class="placeholder-label">${failed ? 'Video unavailable' : 'Video coming soon'}</span><span class="placeholder-title">${escapeHTML(item.title || 'Demonstration')}</span>`;
    slot.append(shade, content);
    slot.setAttribute('role', 'img');
    slot.setAttribute('aria-label', `${item.title}. ${failed ? 'Video unavailable.' : 'Video coming soon.'}${poster ? item.posterIsVideoFrame ? ' Background is a video frame.' : ' Background is a paper illustration.' : ''}`);
  }

  function mountVideo(container, item) {
    const slot = document.createElement('div'); slot.className = 'video-slot';
    container.append(slot);
    const src = safeURL(item.src);
    if (!src) { placeholder(slot, item); return; }
    const video = document.createElement('video');
    video.controls = true; video.playsInline = true; video.preload = 'auto';
    // Muted inline playback allows autoplay without a prior user gesture.
    video.autoplay = true; video.defaultMuted = true; video.muted = true;
    video.loop = Boolean(item.loop);
    video.setAttribute('aria-label', item.title);
    const poster = safeURL(item.poster); if (poster) video.poster = poster;
    video.addEventListener('error', () => placeholder(slot, item, true), { once: true });
    // Captions are optional; set captions to a local .vtt path in content.js.
    if (safeURL(item.captions)) {
      const track = document.createElement('track');
      track.kind = 'captions'; track.src = safeURL(item.captions);
      track.srclang = item.language || 'en'; track.label = item.captionLabel || 'English';
      video.append(track);
    }
    video.src = src; slot.append(video);
  }

  $('#authors').innerHTML = data.project.authors.map(author => {
    const name = escapeHTML(author.name);
    const url = safeURL(author.url);
    const label = url ? `<a href="${escapeHTML(url)}" target="_blank" rel="noopener">${name}</a>` : name;
    return `<span class="author">${label}<sup>${author.affiliations.join(',')}${author.corresponding ? ',*' : ''}</sup></span>`;
  }).join('');
  $('#affiliations').innerHTML = data.project.affiliations.map(affiliation =>
    `<li><a class="affiliation affiliation-${affiliation.id}" href="${escapeHTML(safeURL(affiliation.url))}" target="_blank" rel="noopener" aria-label="${affiliation.id}: ${escapeHTML(affiliation.name)}"><sup>${affiliation.id}</sup><img src="${escapeHTML(safeURL(affiliation.logo))}" alt="${escapeHTML(affiliation.name)}" title="${escapeHTML(affiliation.short)}"></a></li>`
  ).join('');
  $('#bibtex').textContent = data.project.citation;
  const publicURL = value => {
    if (typeof value !== 'string' || !/^https:\/\//i.test(value.trim())) return '';
    try {
      const url = new URL(value);
      if (/^(localhost|127\.|0\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|\[.*\])/.test(url.hostname)) return '';
      return url.href;
    } catch { return ''; }
  };
  ['paper', 'code'].forEach(key => {
    const url = key === 'paper' ? safeURL(data.project[key]) : publicURL(data.project[key]);
    if (!url) return;
    const button = $(`#${key}-link`);
    const link = document.createElement('a');
    link.id = button.id; link.className = button.className; link.innerHTML = button.innerHTML;
    $('.availability', link)?.remove();
    if (key === 'paper') link.classList.add('primary');
    link.href = url; link.target = '_blank'; link.rel = 'noopener';
    button.replaceWith(link);
  });
  if (publicURL(data.project.arxiv)) {
    const link = document.createElement('a');
    link.className = 'button'; link.href = publicURL(data.project.arxiv);
    link.textContent = 'arXiv ↗'; link.target = '_blank'; link.rel = 'noopener';
    $('.hero-links').append(link);
  }
  mountVideo($('#overview-video'), data.overview);
  $$('[data-seek]').forEach(button => button.addEventListener('click', () => {
    const video = $('#overview-video video');
    if (!video) return;
    const seekAndPlay = () => {
      video.currentTime = Number(button.dataset.seek);
      video.play().catch(() => { /* Native controls remain available if playback is blocked. */ });
    };
    if (video.readyState >= 1) seekAndPlay();
    else video.addEventListener('loadedmetadata', seekAndPlay, { once: true });
  }));

  let currentSplit = 'avg';
  let currentMetric = 1;
  const splitLabels = { gen: 'generalization', id: 'in-distribution', avg: 'average' };
  const metricLabels = ['driving score', 'success rate (%)', 'harmonic mean'];
  function drawChart() {
    const experts = data.results.filter(row => row.expert);
    const descriptions = [];
    $('#results-chart').innerHTML = experts.map(base => {
      const ours = data.results.find(row => row.name === `${base.name} + AutoAgent0`);
      const a = base[currentSplit][currentMetric]; const b = ours[currentSplit][currentMetric];
      const gain = b - a;
      descriptions.push(`${base.name}: ${a.toFixed(1)} standalone, ${b.toFixed(1)} with AutoAgent0`);
      return `<div class="chart-group"><div class="chart-label">${escapeHTML(base.name)}<small>+${gain.toFixed(1)} ${currentMetric === 1 ? 'pp' : 'points'}</small></div><div class="chart-bars"><div class="bar-line"><div class="bar-fill" style="--value:${a}%"><span>${a.toFixed(1)}</span></div></div><div class="bar-line"><div class="bar-fill ours" style="--value:${b}%"><span>${b.toFixed(1)}</span></div></div></div></div>`;
    }).join('') + '<div class="chart-axis" aria-hidden="true"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>';
    $('#results-chart').setAttribute('aria-label', `${metricLabels[currentMetric]} on ${splitLabels[currentSplit]} routes. ${descriptions.join('. ')}.`);
    $('#chart-note').textContent = `${currentSplit === 'avg' ? 'Arithmetic mean of both splits' : currentSplit === 'gen' ? 'Matched routes with long-tail distribution shifts' : 'Matched in-distribution base routes'}. Gains are absolute ${currentMetric === 1 ? 'percentage points' : 'score points'} over each standalone expert.`;
  }
  $$('[data-split]').forEach(button => button.addEventListener('click', () => {
    currentSplit = button.dataset.split;
    $$('[data-split]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    drawChart();
  }));
  $('#metric').addEventListener('change', event => { currentMetric = Number(event.target.value); drawChart(); });
  drawChart();
  $('#benchmark-body').innerHTML = data.results.map((row, index) => {
    const base = row.ours ? data.results[index - 1] : null;
    const rgb = '<td class="sensor-cell" aria-label="RGB input: yes">✓</td>';
    const lidar = `<td class="sensor-cell" aria-label="LiDAR input: ${row.input.includes('LiDAR') ? 'yes' : 'no'}">${row.input.includes('LiDAR') ? '✓' : '—'}</td>`;
    const scores = [...row.id, ...row.gen].map(value => `<td>${value.toFixed(1)}</td>`).join('');
    const average = row.avg.map((value, metric) => `<td class="average-cell">${value.toFixed(1)}${base ? `<span class="table-gain">(+${(value - base.avg[metric]).toFixed(1)})</span>` : ''}</td>`).join('');
    return `<tr class="${row.ours ? 'ours-row' : row.expert ? 'expert-row' : ''}"><th scope="row">${escapeHTML(row.name)}</th>${rgb}${lidar}${scores}${average}</tr>`;
  }).join('');
  $('#navsafe-body').innerHTML = data.navsafe.map((row, index) => {
    const base = row[0].includes('AutoAgent0') ? data.navsafe[index - 1] : null;
    const scores = row.slice(1).map((value, metric) => {
      // Table 2 reports gains for DS, SR and ability scores, not DE or comfort.
      const gain = base && ![2, 3].includes(metric)
        ? `<span class="table-gain">(+${(value - base[metric + 1]).toFixed(1)})</span>` : '';
      return `<td>${value.toFixed(1)}${gain}</td>`;
    }).join('');
    return `<tr class="${base ? 'ours-row' : ''}"><th scope="row">${escapeHTML(row[0])}</th>${scores}</tr>`;
  }).join('');

  const comparisonLabels = [
    { key: 'fail1', title: 'End-to-end Method - Fail', caption: '' },
    { key: 'fail2', title: 'Traditional Runtime - Fail', caption: '' },
    { key: 'success', title: 'AutoAgent0 (Ours) - Success', caption: '' }
  ];
  function videoCard(title, caption, item, ours = false) {
    const card = document.createElement('article');
    card.className = `video-card${ours ? ' ours-card' : ''}`;
    const heading = escapeHTML(title)
      .replace(/ - Fail$/, ' - <span class="video-outcome-fail">Fail</span>')
      .replace(/ - Success$/, ' - <span class="video-outcome-success">Success</span>');
    card.innerHTML = `<h4 class="video-card-heading" style="margin:0"><span aria-hidden="true"></span><span>${heading}</span></h4>`;
    mountVideo(card, item);
    return card;
  }
  function showScenario(index) {
    const scenario = data.scenarios[index];
    $$('#scenario-videos video').forEach(video => { video.pause(); video.removeAttribute('src'); video.load(); });
    $('#scenario-title').textContent = scenario.title;
    $('#scenario-counter').textContent = `${String(index + 1).padStart(2, '0')} / ${String(data.scenarios.length).padStart(2, '0')}`;
    $$('#scenario-tabs button').forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
    $('#scenario-videos').replaceChildren(...comparisonLabels.map(label => {
      const configured = scenario.videos[label.key];
      const item = typeof configured === 'string' ? { src: configured } : configured || {};
      return videoCard(label.title, label.caption, { ...item, title: `${scenario.label} · ${label.title}` }, label.key === 'success');
    }));
  }
  data.scenarios.forEach((scenario, index) => {
    const button = document.createElement('button'); button.type = 'button';
    button.textContent = scenario.label; button.setAttribute('aria-pressed', String(index === 0));
    button.addEventListener('click', () => showScenario(index));
    $('#scenario-tabs').append(button);
  });
  if (data.scenarios.length) showScenario(0);
  const navsafeTabs = [];
  const navsafePanels = [];
  function showNavsafe(index) {
    navsafePanels.forEach((panel, i) => {
      const selected = i === index;
      panel.hidden = !selected;
      navsafeTabs[i].setAttribute('aria-selected', String(selected));
      navsafeTabs[i].tabIndex = selected ? 0 : -1;
      if (!selected) $$('video', panel).forEach(video => video.pause());
    });
    const panel = navsafePanels[index];
    const item = data.navsafeCases[index];
    if (!panel.hasChildNodes()) {
      const heading = document.createElement('h3');
      heading.className = 'navsafe-case-title'; heading.textContent = item.title;
      panel.append(heading);
      item.views.forEach(view => {
        const card = document.createElement('article'); card.className = 'video-card navsafe-view';
        const title = document.createElement('h4'); title.className = 'video-card-heading';
        title.textContent = view.title; card.append(title);
        mountVideo(card, { ...view, title: `NavSafe ${item.label} · ${view.title} · SparseDriveV2 and AutoAgent0`, loop: true, posterIsVideoFrame: true });
        panel.append(card);
      });
    } else {
      $$('video', panel).forEach(video => video.play().catch(() => { /* Native controls remain available. */ }));
    }
  }
  data.navsafeCases.forEach((item, index) => {
    const tab = document.createElement('button');
    tab.type = 'button'; tab.id = `navsafe-tab-${item.id}`;
    tab.setAttribute('role', 'tab'); tab.textContent = item.label;
    const panel = document.createElement('div');
    panel.id = `navsafe-panel-${item.id}`; panel.className = 'navsafe-panel'; panel.hidden = true;
    panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', tab.id);
    tab.setAttribute('aria-controls', panel.id);
    tab.addEventListener('click', () => showNavsafe(index));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % navsafeTabs.length;
      else if (event.key === 'ArrowLeft') next = (index + navsafeTabs.length - 1) % navsafeTabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = navsafeTabs.length - 1;
      else return;
      event.preventDefault(); showNavsafe(next); navsafeTabs[next].focus();
    });
    navsafeTabs.push(tab); navsafePanels.push(panel);
    $('#navsafe-tabs').append(tab); $('#navsafe-panels').append(panel);
  });
  if (navsafeTabs.length) showNavsafe(0);

  const runtimeTabs = [];
  const runtimePanels = [];
  function showRuntime(index) {
    runtimePanels.forEach((panel, i) => {
      const selected = i === index;
      if (!selected) $$('video', panel).forEach(video => video.pause());
      panel.hidden = !selected;
      runtimeTabs[i].setAttribute('aria-selected', String(selected));
      runtimeTabs[i].tabIndex = selected ? 0 : -1;
    });
    const panel = runtimePanels[index];
    if (!panel.hasChildNodes()) mountVideo(panel, data.methodVideos[index]);
    else $$('video', panel).forEach(video => {
      video.play().catch(() => { /* Native controls remain available if playback is blocked. */ });
    });
  }
  data.methodVideos.forEach((item, index) => {
    const tab = document.createElement('button');
    tab.type = 'button'; tab.id = `runtime-tab-${item.id}`;
    tab.setAttribute('role', 'tab'); tab.textContent = item.title;
    const panel = document.createElement('div');
    panel.id = `runtime-panel-${item.id}`; panel.className = 'runtime-panel';
    panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', tab.id);
    tab.setAttribute('aria-controls', panel.id);
    tab.addEventListener('click', () => showRuntime(index));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % runtimeTabs.length;
      else if (event.key === 'ArrowLeft') next = (index + runtimeTabs.length - 1) % runtimeTabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = runtimeTabs.length - 1;
      else return;
      event.preventDefault(); showRuntime(next); runtimeTabs[next].focus();
    });
    runtimeTabs.push(tab); runtimePanels.push(panel);
    $('#runtime-tabs').append(tab); $('#method-videos').append(panel);
  });
  if (runtimeTabs.length) showRuntime(0);

  const stages = [
    { kicker: 'SCENE ASSESSMENT', title: 'Determine when recovery is needed.', text: 'Camera, LiDAR, and vision-language reasoning build a clearer view of the scene. The verifier checks the driving expert\'s proposed trajectories for safety and route consistency. If none passes, the runtime starts recovery.', chips: ['Scene context refinement', 'Trajectory verification', 'Recovery trigger'] },
    { kicker: 'RECOVERY MANEUVERS', title: 'Choose a maneuver for the situation.', text: 'The agent uses the scene and the reason for rejection to choose a maneuver: brake, creep forward, bypass an obstacle, or reverse. In this example, clear space to the front-left suggests a bypass.', chips: ['Brake / creep', 'Lateral bypass', 'Reverse'] },
    { kicker: 'VERIFIER FEEDBACK', title: 'Refine the plan before execution.', text: 'The agent adjusts the maneuver\'s path, speed, and duration, then checks it with the same verifier used for the expert\'s plans. Rejection feedback guides another attempt or a different maneuver. Only a plan that passes can be executed.', chips: ['Adjust the maneuver', 'Shared verifier', 'Revise with feedback'] },
  ];
  const methodTabs = $$('.method-steps [role="tab"]');
  const methodPanel = $('#method-panel');
  function showStage(index) {
    const stage = stages[index];
    $('#method-kicker').textContent = stage.kicker; $('#method-title').textContent = stage.title;
    $('#method-text').textContent = stage.text;
    $('#method-chips').innerHTML = stage.chips.map(chip => `<span>${escapeHTML(chip)}</span>`).join('');
    methodTabs.forEach((tab, i) => {
      const selected = i === index;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    methodPanel.setAttribute('aria-labelledby', methodTabs[index].id);
  }
  methodTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => showStage(index));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % methodTabs.length;
      else if (event.key === 'ArrowLeft') next = (index + methodTabs.length - 1) % methodTabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = methodTabs.length - 1;
      else return;
      event.preventDefault(); showStage(next); methodTabs[next].focus();
    });
  });
  showStage(0);

  const dialog = $('#figure-dialog');
  let figureTrigger = null;
  $$('.zoomable').forEach(button => button.addEventListener('click', () => {
    figureTrigger = button;
    $('#enlarged-figure').src = button.dataset.image;
    $('#enlarged-figure').alt = $('img', button).alt;
    $('#figure-caption').textContent = button.dataset.caption;
    dialog.showModal(); document.body.style.overflow = 'hidden';
  }));
  $('#close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => { document.body.style.overflow = ''; if (figureTrigger) figureTrigger.focus(); });

  const themeButton = $('.theme-toggle');
  const darkPreference = window.matchMedia('(prefers-color-scheme: dark)');
  let explicitTheme = false;
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    themeButton.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
    $('meta[name="theme-color"]').content = theme === 'dark' ? '#101620' : '#fbfcfe';
  }
  try {
    const saved = localStorage.getItem('autoagent0-theme');
    explicitTheme = ['dark', 'light'].includes(saved);
    setTheme(explicitTheme ? saved : darkPreference.matches ? 'dark' : 'light');
  } catch { setTheme(darkPreference.matches ? 'dark' : 'light'); }
  themeButton.addEventListener('click', () => {
    explicitTheme = true;
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(next); try { localStorage.setItem('autoagent0-theme', next); } catch { /* File previews may block storage. */ }
  });
  darkPreference.addEventListener('change', event => { if (!explicitTheme) setTheme(event.matches ? 'dark' : 'light'); });

  $('#copy-citation').addEventListener('click', async () => {
    try {
      if (!navigator.clipboard || !window.isSecureContext) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(data.project.citation);
      $('#copy-status').textContent = 'BibTeX copied to clipboard.';
    } catch {
      const selection = window.getSelection(); const range = document.createRange();
      range.selectNodeContents($('#bibtex')); selection.removeAllRanges(); selection.addRange(range);
      let copied = false;
      try { copied = document.execCommand('copy'); } catch { /* Manual copy remains available. */ }
      $('#copy-status').textContent = copied ? 'BibTeX copied to clipboard.' : 'BibTeX selected. Press Ctrl+C (or ⌘C) to copy.';
    }
  });

  const sections = $$('.page > .section');
  const navLinks = $$('.contents nav a');
  let scrollScheduled = false;
  function updateScroll() {
    const height = document.documentElement.scrollHeight - window.innerHeight;
    $('#reading-progress').style.width = `${height > 0 ? Math.min(100, Math.max(0, window.scrollY / height * 100)) : 0}%`;
    let current = '';
    sections.forEach(section => { if (section.getBoundingClientRect().top <= window.innerHeight * .35) current = section.id; });
    navLinks.forEach(link => {
      const active = link.hash === `#${current}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
    scrollScheduled = false;
  }
  function scheduleScroll() { if (!scrollScheduled) { scrollScheduled = true; requestAnimationFrame(updateScroll); } }
  window.addEventListener('scroll', scheduleScroll, { passive: true });
  window.addEventListener('resize', scheduleScroll, { passive: true });
  document.addEventListener('toggle', scheduleScroll, true);
  window.addEventListener('load', updateScroll, { once: true });
  updateScroll();
})();
