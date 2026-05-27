#!/bin/bash
# Watchdog: emails you when all train_iql.py processes finish.
#
# Usage (run in any tmux pane on the GPU node, alongside the salloc shell):
#   bash scripts/email_on_done.sh
#
# Settings: edit MAIL_TO and ATTACH_LOGS_DIR below if needed.

set -e
cd "$(dirname "$0")/.."

MAIL_TO="${MAIL_TO:-alloydas@iastate.edu}"
POLL_SECONDS=60
JOB_PATTERN='train_iql.py'

echo "[watchdog] $(date)  starting — will email $MAIL_TO when no '$JOB_PATTERN' procs remain."
echo "[watchdog] this is non-blocking for the training itself; safe to leave running."

# Wait for jobs to actually be running first (avoid emailing immediately if pgrep returns 0)
sleep 10
INITIAL=$(pgrep -af "$JOB_PATTERN" | grep -v 'watchdog\|email_on_done' | wc -l)
echo "[watchdog] currently $INITIAL training procs alive — will wait until 0."

if [[ "$INITIAL" == "0" ]]; then
    echo "[watchdog] no training procs found — was anything ever launched? exiting."
    exit 1
fi

# Poll
while true; do
    sleep "$POLL_SECONDS"
    n=$(pgrep -af "$JOB_PATTERN" | grep -v 'watchdog\|email_on_done' | wc -l)
    if [[ "$n" == "0" ]]; then
        break
    fi
done

echo "[watchdog] $(date)  all training procs done; preparing email."

# Build the summary body
TMPBODY=$(mktemp)
{
    echo "DiSA-RL training run finished at $(date)."
    echo "Host: $(hostname)"
    echo
    echo "=== Final eval scores per log ==="
    for f in logs/v1_*.log logs/base_*.log; do
        [ -f "$f" ] || continue
        last=$(grep "normalized=" "$f" 2>/dev/null | tail -1)
        best=$(grep "Best normalized" "$f" 2>/dev/null | tail -1)
        if [ -n "$last$best" ]; then
            echo "── $(basename $f)"
            [ -n "$last" ] && echo "   last: $last"
            [ -n "$best" ] && echo "   $best"
        fi
    done
    echo
    echo "=== checkpoint dirs created ==="
    find checkpoints -maxdepth 4 -name 'final.pt' -newer /tmp 2>/dev/null | head -20
    echo
    echo "Logs at: $(pwd)/logs/"
} > "$TMPBODY"

SUBJECT="[DiSA-RL] training run finished on $(hostname)"

# Try `mail` first; fall back to `sendmail` or python smtplib if missing
if command -v mail >/dev/null 2>&1; then
    mail -s "$SUBJECT" "$MAIL_TO" < "$TMPBODY"
    echo "[watchdog] email sent via 'mail' to $MAIL_TO"
elif command -v sendmail >/dev/null 2>&1; then
    {
        echo "Subject: $SUBJECT"
        echo "To: $MAIL_TO"
        echo
        cat "$TMPBODY"
    } | sendmail -t
    echo "[watchdog] email sent via 'sendmail' to $MAIL_TO"
else
    echo "[watchdog] WARN: no 'mail' or 'sendmail' on this node — body saved to logs/done_email.txt"
    cp "$TMPBODY" logs/done_email.txt
fi

rm -f "$TMPBODY"
echo "[watchdog] done."
