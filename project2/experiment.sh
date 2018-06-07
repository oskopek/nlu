#!/usr/bin/env bash
set -e

flags="$1"
new_flags="sct/flags.py"
old_flags="sct/.flags.py.old"

echo "Running experiment `basename "$flags"`..."

mv "$new_flags" "$old_flags"
cp "$flags" "$new_flags"

if ! [ -x "$(command -v bsub)" ]; then
  echo 'bsub is not installed, running locally.' >&2
  python -m sct.train
else
  bsub -W 04:00 -n 4 -R "rusage[mem=4096,ngpus_excl_p=1]" python -m sct.train
fi
echo "Submitted/ran experiment `basename "$flags"`."

