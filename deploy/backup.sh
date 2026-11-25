#!/usr/bin/env bash
# Nightly Clawdia backup, run by clawdia-backup.service as user clawdia.
# Writes /var/backups/clawdia/<YYYY-MM-DD>/{clawdia.sqlite,media/} and prunes
# dated directories so that 14 days remain. Safe to re-run on the same day.
set -euo pipefail
umask 077

STATE_DIR="${CLAWDIA_STATE_DIR:-/var/lib/clawdia}"
BACKUP_ROOT="${CLAWDIA_BACKUP_DIR:-/var/backups/clawdia}"
CLAWDIA_BIN="${CLAWDIA_BIN:-/opt/clawdia/.venv/bin/clawdia}"
KEEP_DAYS=14  # find -mtime +13 matches dirs 14 or more days old, so 14 daily dirs stay

stamp="$(TZ=Europe/Lisbon date +%F)"
dest="$BACKUP_ROOT/$stamp"
work="$BACKUP_ROOT/.partial-$stamp"

rm -rf -- "$work"
mkdir -p -- "$work/media"

"$CLAWDIA_BIN" export-backup --to "$work/clawdia.sqlite" >/dev/null
if [ -d "$STATE_DIR/media" ]; then
  cp -a -- "$STATE_DIR/media/." "$work/media/"
fi

# Swap the finished copy into place only after both parts succeeded.
rm -rf -- "$dest"
mv -- "$work" "$dest"

find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -name '20??-??-??' -mtime "+$((KEEP_DAYS - 1))" -exec rm -rf -- {} +
# Leftovers of a crashed run.
find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -name '.partial-*' -mtime +1 -exec rm -rf -- {} +
echo "backup ok: $dest"
