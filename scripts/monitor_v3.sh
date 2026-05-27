#!/bin/bash
# Background monitor for the 12-job v3 IQL sweep.
# Emits: per-run completion, errors, stalls, and a heartbeat every ~25 min.
# Each stdout line becomes a notification — keep it selective.

cd "$(dirname "$0")/.."
STATE=$(mktemp -d)
CYCLE=0
echo "v3 monitor armed $(date '+%F %T') — watching logs/v3_*.log"

while true; do
    CYCLE=$((CYCLE + 1))
    running=0; done=0; errored=0
    summary=""

    for f in logs/v3_*.log; do
        [ -e "$f" ] || continue
        name=$(basename "$f" .log | sed 's/^v3_//; s/_s0$//; s/-medium//; s/-v2//')

        if grep -qa "Done. Best normalized" "$f" 2>/dev/null; then
            done=$((done + 1))
            if [ ! -e "$STATE/$name.done" ]; then
                touch "$STATE/$name.done"
                echo "[DONE] $name — $(grep -a 'Done. Best' "$f" | tail -1)"
            fi
            continue
        fi

        if grep -qaE "Traceback|CUDA out of memory|RuntimeError|AssertionError|Killed|SystemExit" "$f" 2>/dev/null; then
            errored=$((errored + 1))
            if [ ! -e "$STATE/$name.err" ]; then
                touch "$STATE/$name.err"
                echo "[ERROR] $name — $(grep -aE 'Traceback|CUDA out of memory|RuntimeError|AssertionError|Killed|SystemExit' "$f" | tail -1 | cut -c1-180)"
            fi
            continue
        fi

        running=$((running + 1))
        sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
        prev=$(cat "$STATE/$name.size" 2>/dev/null || echo -1)
        echo "$sz" > "$STATE/$name.size"
        step=$(grep -a 'normalized=' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]{4,7}' | head -1)
        summary="$summary ${name}@${step:-start}"
        if [ "$sz" = "$prev" ] && [ "$CYCLE" -gt 1 ]; then
            echo "[STALL?] $name — log unchanged for ~25 min (size $sz) — may be hung/dead"
        fi
    done

    echo "[HB $(date '+%H:%M')] running=$running done=$done err=$errored |$summary"

    if [ "$running" -eq 0 ] && [ $((done + errored)) -gt 0 ]; then
        echo "[ALL DONE] $(date '+%F %T') — done=$done errored=$errored"
        break
    fi
    sleep 1500
done
