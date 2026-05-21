/* ───────────────────────────────────────────────────────────────────
   citations.js — resolves <span class="cite" data-key="…"></span>
   into numbered author-year citations + an auto-built References
   section with back-references.

   Ported from zblasingame.github.io / assets/js/citations.js, trimmed
   to the keys referenced on this page (Greedy-DiM landing page).
   ─────────────────────────────────────────────────────────────────── */
(function(){
  const C = {
    // ─── face morphing / FR ───
    'ferrara_magicpassport':{authors:'Ferrara, M., Franco, A., Maltoni, D.', title:'The magic passport', venue:'IJCB', year:2014},
    'ferrara_map':       {authors:'Ferrara, M., Franco, A., Maltoni, D.', title:'Morphing Attack Potential', venue:'IWBF', year:2022},
    'mipgan':            {authors:'Zhang, H., Venkatesh, S., Ramachandra, R., Raja, K., Damer, N., Busch, C.', title:'MIPGAN — Generating Strong and High Quality Morphing Attacks Using Identity Prior Driven GAN', venue:'IEEE T-BIOM 3(3)', year:2021, url:'https://arxiv.org/abs/2009.01729'},
    'mmpmr':             {authors:'Scherhag, U., Nautsch, A., Rathgeb, C., et al.', title:'Biometric Systems under Morphing Attacks: Assessment of Morphing Techniques and Vulnerability Reporting', venue:'BIOSIG', year:2017},
    'morphpipe':         {authors:'Zhang, H., Ramachandra, R., Raja, K., Busch, C.', title:'Morph-PIPE: Plugging in Identity Prior to Enhance Face Morphing Attack Based on Diffusion Model', venue:'NIST IBPC', year:2024},
    'synmad':            {authors:'Huber, M., Boutros, F., Luu, A. T., et al.', title:'SYN-MAD 2022: Competition on Face Morphing Attack Detection Based on Privacy-aware Synthetic Training Data', venue:'IJCB', year:2022, url:'https://arxiv.org/abs/2208.07337'},
    'frll':              {authors:'DeBruine, L., Jones, B.', title:'Face Research Lab London Set', venue:'figshare', year:2017, doi:'10.6084/m9.figshare.5047666.v3'},

    // ─── FR systems ───
    'arcface':           {authors:'Deng, J., Guo, J., Xue, N., Zafeiriou, S.', title:'ArcFace: Additive Angular Margin Loss for Deep Face Recognition', venue:'CVPR', year:2019, url:'https://arxiv.org/abs/1801.07698'},
    'adaface':           {authors:'Kim, M., Jain, A. K., Liu, X.', title:'AdaFace: Quality Adaptive Margin for Face Recognition', venue:'CVPR', year:2022, url:'https://arxiv.org/abs/2204.00964'},
    'elasticface':       {authors:'Boutros, F., Damer, N., Kirchbuchner, F., Kuijper, A.', title:'ElasticFace: Elastic Margin Loss for Deep Face Recognition', venue:'CVPRW', year:2022, url:'https://arxiv.org/abs/2109.09416'},

    // ─── diffusion / generative ───
    'blasingame_dim':    {authors:'Blasingame, Z. W., Liu, C.', title:'Leveraging Diffusion for Strong and High Quality Face Morphing Attacks', venue:'IEEE T-BIOM 6(1)', year:2024, doi:'10.1109/TBIOM.2024.3349857'},
    'fast_dim':          {authors:'Blasingame, Z. W., Liu, C.', title:'Fast-DiM: Towards Fast Diffusion Morphs', venue:'IEEE Security & Privacy 22(4)', year:2024, url:'https://arxiv.org/abs/2310.09484'},
    'traveling_salesman':{authors:'Gutin, G., Yeo, A., Zverovich, A.', title:'Traveling salesman should not be greedy: domination analysis of greedy-type heuristics for the TSP', venue:'Discrete Applied Mathematics 117(1)', year:2002, doi:'10.1016/S0166-218X(01)00195-0'},
    'diffae':            {authors:'Preechakul, K., Chatthee, N., Wizadwongsa, S., Suwajanakorn, S.', title:'Diffusion Autoencoders: Toward a Meaningful and Decodable Representation', venue:'CVPR', year:2022, url:'https://arxiv.org/abs/2111.15640'},
    'song2021scorebased':{authors:'Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., Poole, B.', title:'Score-Based Generative Modeling through Stochastic Differential Equations', venue:'ICLR', year:2021, url:'https://arxiv.org/abs/2011.13456'},
    'song2021denoising': {authors:'Song, J., Meng, C., Ermon, S.', title:'Denoising Diffusion Implicit Models', venue:'ICLR', year:2021, url:'https://arxiv.org/abs/2010.02502'},

    // ─── solvers ───
    'lu2023dpmsolver':   {authors:'Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.', title:'DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models', venue:'arXiv:2211.01095', year:2022, url:'https://arxiv.org/abs/2211.01095'},
  };

  /* ─── Sidenote (footnote) markup ─────────────────────────────── */
  let snCounter = 0;
  function makeNote(html){
    snCounter += 1;
    const id = 'sn-' + snCounter;
    const sup  = '<label class="margin-toggle-label" for="' + id + '"><sup class="sidenote-number"></sup></label>';
    const cb   = '<input type="checkbox" id="' + id + '" class="margin-toggle">';
    const note = '<span class="sidenote">' + html + '</span>';
    return sup + cb + note;
  }

  /* ─── Author-year formatting helpers ─────────────────────────── */
  function parseSurnames(s){
    if(!s) return [];
    const parts = s.split(',').map(t => t.trim()).filter(Boolean);
    const out = [];
    for(const p of parts){
      if(/^et\s+al\.?$/i.test(p)) { out.push('et al.'); continue; }
      if(/^([A-Z]\.?\s*\-?\s*)+$/.test(p)) continue;
      out.push(p);
    }
    return out;
  }
  function shortAuthors(s){
    const sur = parseSurnames(s);
    if(!sur.length) return '';
    if(sur.length === 1) return sur[0];
    if(sur[1] === 'et al.') return sur[0] + ' et al.';
    if(sur.length === 2)   return sur[0] + ' and ' + sur[1];
    return sur[0] + ' et al.';
  }

  /* ─── Bibliography entry formatting ──────────────────────────── */
  function fmtBibEntry(key){
    const c = C[key];
    if(!c) return '<em>[missing: ' + key + ']</em>';
    let s = '';
    if(c.authors) s += c.authors;
    if(c.year)    s += ' (' + c.year + ').';
    if(c.title){
      s += ' <em>' + c.title + '</em>';
      if(!/[.!?]$/.test(c.title)) s += '.';
    }
    if(c.venue) s += ' ' + c.venue + '.';
    if(c.url)       s += ' <a href="' + c.url + '">link</a>';
    else if(c.doi)  s += ' <a href="https://doi.org/' + c.doi + '">doi</a>';
    return s;
  }

  /* ─── Main pass ──────────────────────────────────────────────── */
  function process(){
    const cited = Object.create(null);
    let siteCounter = 0;

    document.querySelectorAll('.cite').forEach(el => {
      const keys = (el.getAttribute('data-key') || '').split(',').map(s => s.trim()).filter(Boolean);
      if(!keys.length){ el.remove(); return; }
      const style = (el.getAttribute('data-style') || 'parenthetical').toLowerCase();

      siteCounter += 1;
      const siteId = 'cite-' + siteCounter;

      const parts = keys.map(k => {
        const c = C[k];
        if(!c){
          return '<span class="cite-missing">[missing: ' + k + ']</span>';
        }
        (cited[k] = cited[k] || []).push({id: siteId, idx: siteCounter});
        const auth = shortAuthors(c.authors);
        const year = c.year || 'n.d.';
        const href = '#bib-' + cssEscape(k);
        if(style === 'year' && keys.length === 1){
          return '<a href="' + href + '" class="cite-link">(' + year + ')</a>';
        }
        if(style === 'narrative' && keys.length === 1){
          return '<a href="' + href + '" class="cite-link">' + auth + '</a>\u202F(' + year + ')';
        }
        return '<a href="' + href + '" class="cite-link">' + auth + ', ' + year + '</a>';
      });

      let inner;
      if((style === 'narrative' || style === 'year') && keys.length === 1){
        inner = parts[0];
      } else {
        inner = '(' + parts.join('; ') + ')';
      }
      const html = '<span class="cite-inline" id="' + siteId + '">' + inner + '</span>';

      const tmp = document.createElement('span');
      tmp.innerHTML = html;
      const node = tmp.firstChild;

      const prev = el.previousSibling;
      const needSpace = prev && prev.nodeType === Node.TEXT_NODE && !/\s$/.test(prev.nodeValue)
                      || prev && prev.nodeType === Node.ELEMENT_NODE;
      el.replaceWith(node);
      if(needSpace){
        node.parentNode.insertBefore(document.createTextNode(' '), node);
      }
    });

    document.querySelectorAll('.footnote').forEach(el => {
      const html = el.innerHTML;
      const wrap = document.createElement('span');
      wrap.innerHTML = makeNote(html);
      el.replaceWith(wrap);
    });

    document.querySelectorAll('.margin').forEach(el => {
      el.classList.remove('margin');
      el.classList.add('marginnote');
    });

    buildBibliography(cited);
  }

  function buildBibliography(cited){
    const keys = Object.keys(cited);
    if(!keys.length) return;

    keys.sort((a, b) => {
      const A = (parseSurnames((C[a]||{}).authors)[0] || a).toLowerCase();
      const B = (parseSurnames((C[b]||{}).authors)[0] || b).toLowerCase();
      if(A < B) return -1;
      if(A > B) return  1;
      const yA = (C[a]||{}).year || 0;
      const yB = (C[b]||{}).year || 0;
      return yA - yB;
    });

    // Target container — inside #references if it exists, else appended to body
    let host = document.getElementById('references-list-host');
    if(!host){
      const section = document.createElement('section');
      section.id = 'references';
      section.className = 'references';
      section.setAttribute('aria-label', 'References');
      const wrap = document.createElement('div');
      wrap.className = 'article-wrapper';
      const h2 = document.createElement('h2');
      h2.textContent = 'References';
      wrap.appendChild(h2);
      host = document.createElement('div');
      host.id = 'references-list-host';
      wrap.appendChild(host);
      section.appendChild(wrap);
      document.body.appendChild(section);
    }

    const ol = document.createElement('ol');
    ol.className = 'bib-list';

    keys.forEach(k => {
      const li = document.createElement('li');
      li.id = 'bib-' + cssEscape(k);
      li.className = 'bib-entry';

      const body = document.createElement('span');
      body.className = 'bib-body';
      body.innerHTML = fmtBibEntry(k);
      li.appendChild(body);

      const sites = cited[k];
      if(sites && sites.length){
        const back = document.createElement('span');
        back.className = 'bib-backrefs';
        back.appendChild(document.createTextNode(' ['));
        sites.forEach((s, i) => {
          if(i) back.appendChild(document.createTextNode(', '));
          const a = document.createElement('a');
          a.href = '#' + s.id;
          a.className = 'bib-backref';
          a.textContent = '§' + s.idx;
          back.appendChild(a);
        });
        back.appendChild(document.createTextNode(']'));
        li.appendChild(back);
      }

      ol.appendChild(li);
    });
    // Clear any prior content and append
    while(host.firstChild) host.removeChild(host.firstChild);
    host.appendChild(ol);
  }

  function cssEscape(s){
    if(window.CSS && CSS.escape) return CSS.escape(s);
    return String(s).replace(/[^a-zA-Z0-9_\-]/g, '_');
  }

  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', process);
  } else {
    process();
  }
})();
