"""
Point the whole project at your GitHub repository in one step.

Usage:  python set_repo.py <github-username> [repo-name]

Updates the repository URL used by every data and code link on the website,
rewrites the live-site link in the README, and rebuilds the site.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

if len(sys.argv) < 2:
    sys.exit(__doc__)

user = sys.argv[1].strip().strip("/")
repo = (sys.argv[2] if len(sys.argv) > 2 else "sound-of-success").strip().strip("/")

repo_url = f"https://github.com/{user}/{repo}"
pages_url = f"https://{user.lower()}.github.io/{repo}/"

# --- build_site.py -----------------------------------------------------------
p = os.path.join(ROOT, "docs", "build_site.py")
s = open(p, encoding="utf-8").read()
s = re.sub(r'^REPO = "[^"]*"', f'REPO = "{repo_url}"', s, count=1, flags=re.M)
open(p, "w", encoding="utf-8").write(s)
print(f"  build_site.py  -> {repo_url}")

# --- README ------------------------------------------------------------------
p = os.path.join(ROOT, "README.md")
s = open(p, encoding="utf-8").read()
s = re.sub(r"\*\*Live site:\*\* \S+", f"**Live site:** {pages_url}", s, count=1)
open(p, "w", encoding="utf-8").write(s)
print(f"  README.md      -> {pages_url}")

# --- rebuild -----------------------------------------------------------------
subprocess.run([sys.executable, "build_site.py"],
               cwd=os.path.join(ROOT, "docs"), check=True)

print(f"""
Done. Next, from this folder ({os.path.abspath(ROOT)}):

  git remote add origin {repo_url}.git
  git branch -M main
  git push -u origin main

Then on GitHub: Settings -> Pages -> Source "Deploy from a branch",
branch "main", folder "/docs" -> Save.

Your site will be live at:
  {pages_url}
""")
