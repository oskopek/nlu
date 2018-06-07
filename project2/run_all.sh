#!/usr/bin/env bash
set -e

new_flags="sct/flags.py"
last_name=""

for file in `ls sct/experiments/`; do
    if [[ $file != roemmele* ]]; then
        continue
    fi
    flags="sct/experiments/$file"
    if [[ -f $flags ]]; then
        expname="`basename $flags`"
        echo "Running experiment $expname..."
        if [[ -z $last_name ]]; then
            bsub -J "$expname" -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" "cp "$flags" "$new_flags" && python -m sct.train $@"
        else
            bsub -J "$expname" -w 'ended('"$last_name"')' -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" "cp "$flags" "$new_flags" && python -m sct.train $@"
        fi
        last_name="$expname"
        echo "Submitted experiment $expname."
        echo
    fi
done


