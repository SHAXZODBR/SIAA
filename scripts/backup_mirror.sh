#!/usr/bin/env bash
# ==============================================================================
#   SENTINEL — MIRROR THE DATA ARCHIVE A → B (second drive), WITH VERIFICATION
# ==============================================================================
#   WHAT
#     "When a drive arrives" step 2 (docs/release_runbook_en.md §2a): after
#     scripts/intake_batch.py has archived a batch on the primary archive drive A,
#     make the backup drive B an exact copy of A:
#
#       1. refuse when B looks NEWER than A (B's raw/INVENTORY.csv is newer than
#          A's, or B has an inventory and A has none) — that means the drives are
#          swapped or a batch was archived on B only; mirroring would delete it;
#       2. refuse when A is empty (an unmounted / wrong A would wipe B);
#       3. DRY RUN: rsync -a --checksum --delete-after --dry-run --itemize-changes
#          → how many files would be copied / updated / DELETED on B (the first
#          deletions are listed);
#       4. confirm: --yes, or an interactive "type yes" prompt (no terminal and
#          no --yes → refused);
#       5. the real mirror: rsync -a --checksum --delete-after A/ B/
#          (macOS junk — .DS_Store, ._*, .Spotlight-V100, .Trashes, .fseventsd —
#          is neither copied nor deleted);
#       6. sample verification: N random files of A (default 50) are hashed with
#          `shasum -a 256` on BOTH sides and compared;
#       7. summary + exit code.
#
#   USAGE
#     scripts/backup_mirror.sh [--yes] [--dry-run] [--verify-only] [--sample N] <A> <B>
#       A, B          archive roots (e.g. /Volumes/SIAA_DATA /Volumes/SIAA_BACKUP); both must
#                     exist — B is never created (an unmounted /Volumes/X would land on the
#                     internal disk)
#       --yes         do not ask before the real mirror
#       --dry-run     stop after the dry run (changes nothing)
#       --verify-only skip rsync; only the sample verification (e.g. a monthly check)
#       --sample N    files to verify (default 50; 0 = skip)
#
#   EXIT CODES
#     0 ok · 1 usage / refused / not confirmed · 2 rsync failed · 3 sample verification failed
#
#   PHI: only counts and archive-relative paths of deleted / differing files are
#   printed, and a file under a reports/ folder (named <patientID>_<Name>.pdf) is
#   shown with its name replaced by sha256(name)[:16]. The archive itself holds
#   identifiable data — both drives must be encrypted.
# ==============================================================================
set -euo pipefail

EXIT_OK=0
EXIT_USAGE=1
EXIT_RSYNC=2
EXIT_VERIFY=3

YES=0
DRY_ONLY=0
VERIFY_ONLY=0
SAMPLE=50
RSYNC_EXCLUDES=(--exclude=.DS_Store '--exclude=._*' --exclude=.Spotlight-V100 --exclude=.Trashes
                --exclude=.fseventsd --exclude=.TemporaryItems)

if [ -t 1 ]; then R=$'\033[0;31m'; G=$'\033[0;32m'; Y=$'\033[1;33m'; B=$'\033[0;34m'; NC=$'\033[0m'
else R=''; G=''; Y=''; B=''; NC=''; fi

die() { echo "${R}✗ $*${NC}" >&2; exit "$EXIT_USAGE"; }
usage() { sed -n '/^#   USAGE/,/^#   EXIT CODES/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

POS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --dry-run) DRY_ONLY=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --sample)
      [ $# -ge 2 ] || die "--sample needs a number"
      SAMPLE="$2"; shift ;;
    --sample=*) SAMPLE="${1#--sample=}" ;;
    -h|--help) usage; exit "$EXIT_OK" ;;
    --*) usage >&2; die "unknown option $1" ;;
    *) POS="${POS}${POS:+$'\n'}$1" ;;
  esac
  shift
done
case "$SAMPLE" in ''|*[!0-9]*) die "--sample must be a non-negative integer" ;; esac
N_POS=$(printf '%s' "$POS" | grep -c '' || true)
[ "$N_POS" -eq 2 ] || { usage >&2; die "expected exactly two paths: <A> <B>"; }
A=$(printf '%s\n' "$POS" | sed -n 1p)
BDIR=$(printf '%s\n' "$POS" | sed -n 2p)

for tool in rsync shasum awk; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool not found on PATH"
done
[ -d "$A" ] || die "A ($A) does not exist or is not a directory — is the archive drive mounted?"
[ -d "$BDIR" ] || die "B ($BDIR) does not exist — mount the backup drive (or mkdir the target folder on it); it is never created automatically"
A=$(cd "$A" && pwd -P)
BDIR=$(cd "$BDIR" && pwd -P)
[ "$A" != "$BDIR" ] || die "A and B are the same directory"
case "$BDIR/" in "$A"/*) die "B ($BDIR) lies inside A ($A)" ;; esac
case "$A/" in "$BDIR"/*) die "A ($A) lies inside B ($BDIR)" ;; esac

# --- 1/2. direction + emptiness guards -------------------------------------------------
A_INV="$A/raw/INVENTORY.csv"
B_INV="$BDIR/raw/INVENTORY.csv"
if [ -z "$(ls -A "$A" 2>/dev/null | grep -v -e '^\.DS_Store$' -e '^\.Spotlight-V100$' -e '^\.Trashes$' -e '^\.fseventsd$' || true)" ]; then
  die "A ($A) is empty — refusing to mirror an empty archive over B (wrong or unmounted drive?)"
fi
if [ -f "$B_INV" ]; then
  if [ ! -f "$A_INV" ]; then
    die "B holds raw/INVENTORY.csv but A does not — B looks like the newer archive (drives swapped?). Nothing was changed."
  fi
  if [ "$B_INV" -nt "$A_INV" ]; then
    die "B's raw/INVENTORY.csv is NEWER than A's — a batch may have been archived on B only, or the drives are swapped. Mirroring would delete it. Check both inventories (diff \"$A_INV\" \"$B_INV\"); nothing was changed."
  fi
fi
[ -f "$A_INV" ] || echo "${Y}⚠ A has no raw/INVENTORY.csv (no batch archived with intake_batch.py yet)${NC}"

count_files() {  # regular, non-junk files under $1
  find "$1" -type f ! -name '.DS_Store' ! -name '._*' ! -path '*/.Spotlight-V100/*' ! -path '*/.Trashes/*' \
       ! -path '*/.fseventsd/*' ! -path '*/.TemporaryItems/*' 2>/dev/null | wc -l | tr -d ' '
}

sha_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }

safe_path() {  # archive-relative path for messages: report file names hashed (PHI)
  local p="$1" base dir ext
  case "/$p" in
    */) printf '%s\n' "$p" ;;
    */reports/*)
      base=${p##*/}; dir=${p%/*}; ext=""
      case "$base" in *.*) ext=".${base##*.}" ;; esac
      printf '%s/%s%s\n' "$dir" "$(printf '%s' "$base" | shasum -a 256 | cut -c1-16)" "$ext" ;;
    *) printf '%s\n' "$p" ;;
  esac
}

# --- 6. sample verification -------------------------------------------------------------
verify_sample() {
  local n="$1" listing picked ok=0 bad=0 rel ha hb
  [ "$n" -gt 0 ] || { echo "  sample verification skipped (--sample 0)"; return 0; }
  listing=$(find "$A" -type f ! -name '.DS_Store' ! -name '._*' ! -path '*/.Spotlight-V100/*' ! -path '*/.Trashes/*' \
            ! -path '*/.fseventsd/*' ! -path '*/.TemporaryItems/*' 2>/dev/null \
            | awk -v p="$A/" '{ print substr($0, length(p) + 1) }')
  if [ -z "$listing" ]; then echo "  sample verification: A has no files"; return 0; fi
  picked=$(printf '%s\n' "$listing" | awk 'BEGIN{srand()} {printf "%.12f\t%s\n", rand(), $0}' \
           | sort -k1,1 | head -n "$n" | cut -f2-)
  echo "${B}── sample verification: sha256 of $(printf '%s\n' "$picked" | grep -c '') random files on both sides${NC}"
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    ha=$(sha_of "$A/$rel")
    if [ ! -f "$BDIR/$rel" ]; then
      echo "${R}  ✗ missing on B: $(safe_path "$rel")${NC}"; bad=$((bad + 1)); continue
    fi
    hb=$(sha_of "$BDIR/$rel")
    if [ -n "$ha" ] && [ "$ha" = "$hb" ]; then ok=$((ok + 1))
    else echo "${R}  ✗ sha256 differs: $(safe_path "$rel")${NC}"; bad=$((bad + 1)); fi
  done <<EOF_PICKED
$picked
EOF_PICKED
  VERIFIED_OK=$ok
  VERIFIED_BAD=$bad
  [ "$bad" -eq 0 ]
}

VERIFIED_OK=0
VERIFIED_BAD=0
echo "${B}mirror A → B${NC}"
echo "  A: $A ($(count_files "$A") files)"
echo "  B: $BDIR ($(count_files "$BDIR") files)"

if [ "$VERIFY_ONLY" -eq 1 ]; then
  if verify_sample "$SAMPLE"; then
    echo "${G}✓ sample verification passed: $VERIFIED_OK/$VERIFIED_OK files identical on A and B${NC}"
    exit "$EXIT_OK"
  fi
  echo "${R}✗ SAMPLE VERIFICATION FAILED: $VERIFIED_BAD of $((VERIFIED_OK + VERIFIED_BAD)) files differ — run the mirror again, then check drive B (Disk Utility → First Aid)${NC}"
  exit "$EXIT_VERIFY"
fi

# --- 3. dry run ---------------------------------------------------------------------------
echo "${B}── dry run (nothing is changed yet)${NC}"
if ! PLAN=$(rsync -a --checksum --delete-after --dry-run --itemize-changes "${RSYNC_EXCLUDES[@]}" "$A/" "$BDIR/" 2>&1); then
  printf '%s\n' "$PLAN" | tail -n 20 >&2
  echo "${R}✗ rsync dry run failed${NC}" >&2
  exit "$EXIT_RSYNC"
fi
N_COPY=$(printf '%s\n' "$PLAN" | grep -Ec '^[<>c]f' || true)
N_DEL=$(printf '%s\n' "$PLAN" | grep -c '^\*deleting' || true)
echo "  files to copy / update on B: $N_COPY"
echo "  files/folders to DELETE on B: $N_DEL"
if [ "$N_DEL" -gt 0 ]; then
  printf '%s\n' "$PLAN" | grep '^\*deleting' | head -n 20 | sed 's/^\*deleting *//' \
    | while IFS= read -r p; do echo "    - $(safe_path "$p")"; done
  [ "$N_DEL" -le 20 ] || echo "    … and $((N_DEL - 20)) more"
fi
if [ "$DRY_ONLY" -eq 1 ]; then
  echo "${G}✓ dry run only (--dry-run): nothing was changed${NC}"
  exit "$EXIT_OK"
fi

# --- 4. confirm ---------------------------------------------------------------------------
if [ "$YES" -ne 1 ]; then
  if [ ! -t 0 ]; then
    die "not confirmed: no terminal to ask — re-run with --yes after reading the dry run above"
  fi
  printf '%s' "Mirror A → B now ($N_COPY to copy, $N_DEL to delete on B)? Type yes: "
  read -r ANSWER || ANSWER=""
  [ "$ANSWER" = "yes" ] || die "not confirmed — nothing was changed"
fi

# --- 5. mirror ----------------------------------------------------------------------------
echo "${B}── mirroring …${NC}"
START=$(date +%s)
if ! rsync -a --checksum --delete-after "${RSYNC_EXCLUDES[@]}" "$A/" "$BDIR/"; then
  echo "${R}✗ rsync failed — B may be partially updated; fix the cause (space, permissions, drive) and run again${NC}" >&2
  exit "$EXIT_RSYNC"
fi
ELAPSED=$(( $(date +%s) - START ))

# --- 6. verify + 7. summary ---------------------------------------------------------------
STATUS=$EXIT_OK
verify_sample "$SAMPLE" || STATUS=$EXIT_VERIFY
echo
echo "${B}═══ mirror summary ═══${NC}"
echo "  A                : $A ($(count_files "$A") files)"
echo "  B                : $BDIR ($(count_files "$BDIR") files)"
echo "  copied / updated : $N_COPY"
echo "  deleted on B     : $N_DEL"
echo "  rsync time       : ${ELAPSED}s"
if [ "$SAMPLE" -gt 0 ]; then
  echo "  sample verified  : $VERIFIED_OK identical, $VERIFIED_BAD differing"
fi
if [ "$STATUS" -eq "$EXIT_OK" ]; then
  echo "${G}✓ B is a verified mirror of A${NC}"
else
  echo "${R}✗ SAMPLE VERIFICATION FAILED — do not rely on B; run the mirror again and check drive B${NC}"
fi
exit "$STATUS"
