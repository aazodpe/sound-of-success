"""
Builds the project website from content fragments.

Every page shares one masthead, one tab bar and one footer, so the navigation
only has to be correct in this file. To edit a page, edit its fragment in
content/ and re-run:  python build_site.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(HERE, "content")

SITE_TITLE = "The Sound of Success"
TAGLINE = ("What popular music is actually made of, and whether a song's "
           "success can be seen in the music itself.")
REPO = "https://github.com/aazodpe/sound-of-success"   # <- edit once, applies everywhere

EYEBROW = "CSCI 5612&nbsp;&middot; Machine Learning for DTSC&nbsp;&middot; Website Project"

BYLINE = "Atharva Zodpe"

# (output filename, tab label, fragment filename)
# Tab labels follow the names required by the assignment.
PAGES = [
    ("index.html",        "Introduction",  "introduction.html"),
    ("dataprep_eda.html", "DataPrep_EDA",  "dataprep_eda.html"),
    ("clustering.html",   "Clustering",    "clustering.html"),
    ("pca.html",          "PCA",           "pca.html"),
    ("naivebayes.html",   "NaiveBayes",    "naivebayes.html"),
    ("dectrees.html",     "DecTrees",      "dectrees.html"),
    ("svms.html",         "SVMs",          "svms.html"),
    ("regression.html",   "Regression",    "regression.html"),
    ("nn.html",           "NN",            "nn.html"),
    ("conclusions.html",  "Conclusions",   "conclusions.html"),
]

WAVE_HEIGHTS = [18, 34, 52, 28, 44, 62, 30, 48, 22, 56, 38, 68, 26, 46, 58,
                32, 50, 20, 42, 64, 36, 54, 24, 40, 60, 30, 48, 34, 22, 44,
                56, 28, 50, 38, 66, 26, 46, 32, 52, 20]


def waveform():
    """A row of bars, staggered so the pulse reads as a moving waveform."""
    bars = []
    for i, h in enumerate(WAVE_HEIGHTS):
        bars.append(f'<i style="height:{h}px;animation-delay:{(i % 12) * 0.13:.2f}s"></i>')
    return '<div class="waveform" aria-hidden="true">' + "".join(bars) + "</div>"


# Applies the saved or system theme before first paint, so there is no flash.
THEME_BOOT = """<script>
(function () {
  try {
    var saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.setAttribute('data-theme', saved);
    }
  } catch (e) {}
})();
</script>"""

THEME_SCRIPT = """<script>
(function () {
  var root = document.documentElement;

  function current() {
    var set = root.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }

  // Every figure ships a light and a dark rendering; show the right one.
  function swapFigures(mode) {
    var imgs = document.querySelectorAll('img[data-dark]');
    for (var i = 0; i < imgs.length; i++) {
      var want = mode === 'dark' ? imgs[i].dataset.dark : imgs[i].dataset.light;
      if (imgs[i].getAttribute('src') !== want) imgs[i].setAttribute('src', want);
    }
  }

  function apply(mode) {
    root.setAttribute('data-theme', mode);
    try { localStorage.setItem('theme', mode); } catch (e) {}
    swapFigures(mode);
  }

  swapFigures(current());

  var btn = document.querySelector('.theme-toggle');
  if (btn) {
    btn.addEventListener('click', function () {
      apply(current() === 'dark' ? 'light' : 'dark');
    });
  }

  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function () {
    if (!localStorage.getItem('theme')) swapFigures(current());
  });

  // Gentle reveal as sections come into view.
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px' });
    document.querySelectorAll('main > figure, main > .cards, main > table').forEach(function (el) {
      el.classList.add('reveal');
      io.observe(el);
    });
  }
})();
</script>"""

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGETITLE__ &middot; __SITETITLE__</title>
<meta name="description" content="__TAGLINE__">
<link rel="preload" href="fonts/inter.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="fonts/instrument-serif.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="style.css">
__THEMEBOOT__
</head>
<body>

<header class="masthead">
  <p class="eyebrow">__EYEBROW__</p>
  <h1>__SITETITLE__</h1>
  <p class="tagline">__TAGLINE__</p>
  __WAVEFORM__
</header>

<nav class="tabs">
  <div class="bar">
    <ul>
__NAV__
    </ul>
    <button class="theme-toggle" type="button" aria-label="Switch between light and dark">
      <span class="moon">&#9789;</span><span class="sun">&#9728;</span>
    </button>
  </div>
</nav>

<main>
__BODY__
</main>

<footer>
  <p>__BYLINE__</p>
</footer>

__THEMESCRIPT__
</body>
</html>
"""


def add_theme_images(body):
    """
    Give every figure image its light and dark source.

    Content fragments only ever reference images/whatever.png. Each figure is
    rendered twice by the plotting scripts, so this adds the data attributes
    the theme script needs, and the dark file becomes the default src when one
    exists (dark is the designed default).
    """
    def repl(m):
        light = m.group(1)
        stem, _, ext = light.rpartition(".")
        dark = f"{stem}_dark.{ext}"
        if not os.path.exists(os.path.join(HERE, dark)):
            return m.group(0)
        return (f'src="{dark}" data-light="{light}" data-dark="{dark}" '
                f'loading="lazy" decoding="async"')

    return re.sub(r'src="(images/[^"]+)"', repl, body)


def build():
    os.makedirs(CONTENT, exist_ok=True)
    for out_name, label, fragment in PAGES:
        nav_items = []
        for other_out, other_label, _ in PAGES:
            css = ' class="active"' if other_out == out_name else ""
            nav_items.append(f'    <li><a href="{other_out}"{css}>{other_label}</a></li>')

        frag_path = os.path.join(CONTENT, fragment)
        if os.path.exists(frag_path):
            with open(frag_path, encoding="utf-8") as fh:
                body = fh.read()
        else:
            body = (f"<h2>{label}</h2>\n"
                    f'<div class="pending"><strong>Coming in a later module.</strong>'
                    f"This section will be filled in as the project progresses.</div>")

        body = add_theme_images(body)

        html = (SHELL
                .replace("__SITETITLE__", SITE_TITLE)
                .replace("__TAGLINE__", TAGLINE)
                .replace("__BYLINE__", BYLINE)
                .replace("__PAGETITLE__", label)
                .replace("__EYEBROW__", EYEBROW)
                .replace("__WAVEFORM__", waveform())
                .replace("__THEMEBOOT__", THEME_BOOT)
                .replace("__THEMESCRIPT__", THEME_SCRIPT)
                .replace("__NAV__", "\n".join(nav_items))
                .replace("__BODY__", body)
                .replace("__REPO__", REPO))

        with open(os.path.join(HERE, out_name), "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"  built {out_name:<20} ({label})")


if __name__ == "__main__":
    print("Building site...")
    build()
    print("Done.")
