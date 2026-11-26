#!/bin/sh
# Build, commit the tracked changes, push, and wait until GitHub Pages serves the new build.
#
#   tools/deploy.sh "Commit message"
#
# New files must be `git add`ed first; only tracked files are committed. The wait checks the live
# page's build stamp (footer) every 10 s for up to 4 minutes, printing as it goes, so it never hangs.
set -e
cd "$(dirname "$0")/.."
[ -n "$1" ] || { echo "usage: tools/deploy.sh \"commit message\"" >&2; exit 2; }

python3 -I tools/build.py
stamp=$(sed -n 's/.*id="build" title="Build of this page">\([0-9a-f]*\)<.*/\1/p' index.html)

if git diff --quiet && git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -qam "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
fi
git push -q origin HEAD
echo "pushed $(git rev-parse --short HEAD); waiting for build $stamp on Pages"

url="https://grazzini.github.io/vocab-drill/"
i=0
while [ $i -lt 24 ]; do
  live=$(curl -fs "$url?deploy=$(date +%s)" | sed -n 's/.*id="build" title="Build of this page">\([0-9a-f]*\)<.*/\1/p') || live=""
  if [ "$live" = "$stamp" ]; then echo "live: $stamp"; exit 0; fi
  echo "  $((i * 10))s: live is ${live:-unreachable}"
  i=$((i + 1))
  sleep 10
done
echo "not live after 4 minutes (live: ${live:-unreachable}); check https://github.com/grazzini/vocab-drill/actions" >&2
exit 1
