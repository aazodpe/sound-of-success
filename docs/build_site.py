"""
Builds the project website from content fragments.

Every page shares one masthead, one tab bar and one footer, so the navigation
only has to be correct in this file. To edit a page, edit its fragment in
content/ and re-run:  python build_site.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(HERE, "content")

SITE_TITLE = "The Sound of Success"
TAGLINE = ("What popular music is actually made of, and whether a song's "
           "success can be seen in the music itself.")
REPO = "https://github.com/AtharvaZodpe/sound-of-success"   # <- edit once, applies everywhere

BYLINE = "Atharva Zodpe &middot; CSCI 5612 Machine Learning &middot; University of Colorado Boulder"

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

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGETITLE__ &middot; __SITETITLE__</title>
<meta name="description" content="__TAGLINE__">
<link rel="stylesheet" href="style.css">
</head>
<body>

<header class="masthead">
  <h1>__SITETITLE__</h1>
  <p class="tagline">__TAGLINE__</p>
  <p class="byline">__BYLINE__</p>
</header>

<nav class="tabs">
  <ul>
__NAV__
  </ul>
</nav>

<main>
__BODY__
</main>

<footer>
  <p>__BYLINE__</p>
</footer>

</body>
</html>
"""


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

        html = (SHELL
                .replace("__SITETITLE__", SITE_TITLE)
                .replace("__TAGLINE__", TAGLINE)
                .replace("__BYLINE__", BYLINE)
                .replace("__PAGETITLE__", label)
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
