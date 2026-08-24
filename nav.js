/* ─────────────────────────────────────────────────────────────
   Shared site-nav behavior — mobile menu, Company/Briefings
   dropdowns, and the wordmark scroll-collapse (full "SAFETY
   CRITICAL LABS" → "SCL" past 40px). Loaded on every page so the
   header behaves identically to the homepage. Pair with the shared
   shell CSS in styles.css. Pages must NOT also bind these inline
   (that would double-fire the toggles).
   ───────────────────────────────────────────────────────────── */
(function () {
  /* Mobile hamburger menu */
  var ham = document.querySelector('.gn-hamburger');
  var links = document.querySelector('.gn-links');
  if (ham && links) {
    ham.addEventListener('click', function () {
      this.classList.toggle('open');
      links.classList.toggle('open');
      this.setAttribute('aria-expanded', this.classList.contains('open'));
      document.body.style.overflow = this.classList.contains('open') ? 'hidden' : '';
    });
  }

  /* Mobile menu CTA: the bar's "Get Certified" pill is hidden at phone
     widths, so clone it in as the menu's last item. Cloned here (one
     shared file) instead of editing every page's nav markup; CSS shows
     .gn-menu-cta-item only inside the ≤880px menu. */
  var cta = document.querySelector('.global-nav .gn-cta');
  if (cta && links && !links.querySelector('.gn-cta')) {
    var ctaItem = document.createElement('li');
    ctaItem.className = 'gn-menu-cta-item';
    ctaItem.appendChild(cta.cloneNode(true));
    links.appendChild(ctaItem);
  }

  /* Dropdown menus (Company / Briefings) — aria-expanded tracks the open
     state so screen readers hear it, and Escape closes + refocuses the
     toggle so keyboard users aren't stranded in a vanished menu. */
  function closeDropdowns() {
    document.querySelectorAll('.gn-dropdown').forEach(function (d) {
      d.classList.remove('open');
      var m = d.querySelector('.gn-dropdown-menu');
      if (m) m.style.display = 'none';
      var t = d.querySelector('.gn-dropdown-toggle');
      if (t) t.setAttribute('aria-expanded', 'false');
    });
  }
  document.querySelectorAll('.gn-dropdown-toggle').forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      var dd = this.closest('.gn-dropdown');
      var menu = dd.querySelector('.gn-dropdown-menu');
      var isOpen = dd.classList.contains('open');
      closeDropdowns();
      if (!isOpen) {
        dd.classList.add('open');
        menu.style.display = 'block';
        this.setAttribute('aria-expanded', 'true');
      }
    });
  });
  document.addEventListener('click', closeDropdowns);
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    var open = document.querySelector('.gn-dropdown.open');
    if (!open) return;
    var t = open.querySelector('.gn-dropdown-toggle');
    closeDropdowns();
    if (t) t.focus();
  });

  /* Wordmark scroll-collapse: full name at top, "SCL" past 40px */
  var nav = document.querySelector('.global-nav');
  if (nav) {
    var THRESHOLD = 40, compact = false;
    var update = function () {
      var should = window.scrollY > THRESHOLD;
      if (should !== compact) { compact = should; nav.classList.toggle('gn-compact', compact); }
    };
    window.addEventListener('scroll', update, { passive: true });
    update();
  }
})();
