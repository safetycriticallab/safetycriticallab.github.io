/* Footer behavior: none. The phone accordion was removed 2026-07-17 in favor
   of the always-open two-column layout (the ≤900px grid in styles.css now
   carries down to phones). This stub stays because every page includes it. */

/* Hero cursor glow — shared driver for the support pages' .inner-hero
   radial highlight (promoted 2026-08-24 from per-page copies). No-op on
   pages without the glow markup. */
(function () {
  var glow = document.querySelector('.fw-hero-glow');
  var hero = document.querySelector('.inner-hero');
  if (!glow || !hero) return;
  hero.addEventListener('mousemove', function (e) {
    var rect = hero.getBoundingClientRect();
    glow.style.setProperty('--glow-x', (e.clientX - rect.left) + 'px');
    glow.style.setProperty('--glow-y', (e.clientY - rect.top) + 'px');
  });
})();
