#!/bin/sh
set -eu

if [ "$(basename "$PWD")" = "script" ]; then
    cd ../
fi


# -------------------------------
# Logging
# -------------------------------
LOG_LEVEL="${LOG_LEVEL:-INFO}"   # DEBUG | INFO | WARN | ERROR
LOG_FILE="${LOG_FILE:-}"         # optional path
VERBOSE="${VERBOSE:-0}"          # 1 => per-file logs

_level_num() {
  case "$1" in
    DEBUG) echo 10 ;;
    INFO)  echo 20 ;;
    WARN)  echo 30 ;;
    ERROR) echo 40 ;;
    *)     echo 20 ;;
  esac
}

_log() {
  level="$1"
  shift
  msg="$*"
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  if [ "$(_level_num "$level")" -lt "$(_level_num "$LOG_LEVEL")" ]; then
    return 0
  fi

  printf "%s [%s] %s\n" "$ts" "$level" "$msg" >&2

  if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    printf "%s [%s] %s\n" "$ts" "$level" "$msg" >>"$LOG_FILE"
  fi
}

log_debug(){ _log DEBUG "$*"; }
log_info(){  _log INFO  "$*"; }
log_warn(){  _log WARN  "$*"; }
log_error(){ _log ERROR "$*"; }

die() { log_error "$*"; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

# -------------------------------
# Config (override via env vars)
# -------------------------------
PKG_DIR="${PKG_DIR:-brmspy}"
DOCS_DIR="${DOCS_DIR:-docs}"
INTERNALS_SUBDIR="${INTERNALS_SUBDIR:-internals}"
INTERNALS_DIR="${DOCS_DIR}/${INTERNALS_SUBDIR}"
NAV_OUT="${NAV_OUT:-${INTERNALS_DIR}/_nav.generated.yml}"
ALLOW_DIRS="${ALLOW_DIRS:-}"  # space-separated, e.g. "brms _build _runtime"

need python3
[ -d "$PKG_DIR" ] || die "PKG_DIR not found: $PKG_DIR"

mkdir -p "$INTERNALS_DIR"

log_info "Starting internals doc generation"
log_info "PKG_DIR=$PKG_DIR DOCS_DIR=$DOCS_DIR INTERNALS_DIR=$INTERNALS_DIR NAV_OUT=$NAV_OUT"
[ -n "$ALLOW_DIRS" ] && log_info "ALLOW_DIRS=$ALLOW_DIRS"
[ -n "$LOG_FILE" ] && log_info "LOG_FILE=$LOG_FILE"
log_info "LOG_LEVEL=$LOG_LEVEL VERBOSE=$VERBOSE"

tmp_scan="$(mktemp)"
tmp_list="$(mktemp)"
nav_items="$(mktemp)"
cleanup() { rm -f "$tmp_scan" "$tmp_list" "$nav_items"; }
trap cleanup EXIT INT TERM

log_info "Scanning for files: ${PKG_DIR}/**/_*.py (excluding __*.py and __init__.py)"
find "$PKG_DIR" -type f -name "*.py" ! -name "__*.py" > "$tmp_scan"


count_scanned=0
count_selected=0

while IFS= read -r f; do
  count_scanned=$((count_scanned + 1))

  rel="${f#"$PKG_DIR"/}"
  base="$(basename "$rel")"

  case "$base" in
    __*) continue ;;
    __init__.py) continue ;;
  esac

  case "$rel" in
    _*|*/_*) : ;;     # keep
    *) continue ;;    # skip
  esac

  if [ -n "$ALLOW_DIRS" ]; then
    top="${rel%%/*}"
    ok=0
    for d in $ALLOW_DIRS; do
      if [ "$top" = "$d" ]; then ok=1; break; fi
    done
    [ "$ok" -eq 1 ] || continue
  fi

  printf "%s\n" "$rel" >> "$tmp_list"
  count_selected=$((count_selected + 1))
  [ "$VERBOSE" = "1" ] && log_debug "Selected: $rel"
done < "$tmp_scan"

[ -s "$tmp_list" ] || { log_warn "No matching files found."; exit 0; }

sort -u "$tmp_list" -o "$tmp_list"
unique_count="$(wc -l <"$tmp_list" | tr -d ' ')"
log_info "Scan complete: scanned=$count_scanned selected=$count_selected unique=$unique_count"

write_template() {
  module="$1"
  cat <<EOF
::: ${module}
    options:
      show_root_heading: false
      show_source: true
      heading_level: 2
      members: true
      filters: []
      show_if_no_docstring: true
EOF
}

count_written=0
while IFS= read -r rel; do
  rel_no_ext="${rel%.py}"                       # brms/_brms_module
  module_suffix="$(printf "%s" "$rel_no_ext" | tr '/' '.')"
  module="${PKG_DIR}.${module_suffix}"          # brmspy.brms._brms_module

  md_rel="${rel_no_ext}.md"                     # brms/_brms_module.md
  md_path="${INTERNALS_DIR}/${md_rel}"          # docs/internals/brms/_brms_module.md

  mkdir -p "$(dirname "$md_path")"
  write_template "$module" > "$md_path"

  count_written=$((count_written + 1))
  printf "%s|%s/%s\n" "$rel_no_ext" "$INTERNALS_SUBDIR" "$md_rel" >> "$nav_items"

  if [ "$VERBOSE" = "1" ]; then
    log_info "Wrote: $md_path (module: $module)"
  fi
done < "$tmp_list"

log_info "Markdown generation complete: wrote=$count_written"
log_info "Generating nav YAML: $NAV_OUT"

python3 - "$nav_items" "$NAV_OUT" <<'PY'
import sys
from collections import OrderedDict

items_path, nav_out = sys.argv[1], sys.argv[2]

items = []
with open(items_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rel_no_ext, docs_rel = line.split("|", 1)
        parts = rel_no_ext.split("/")
        dirs, leaf = parts[:-1], parts[-1]
        items.append((dirs, leaf, docs_rel))

root = OrderedDict()
for dirs, leaf, docs_rel in sorted(items, key=lambda x: (x[0], x[1])):
    node = root
    for d in dirs:
        node = node.setdefault(d, OrderedDict())
    node[leaf] = docs_rel

def emit(node, indent):
    out = []
    for k, v in node.items():
        if isinstance(v, OrderedDict):
            out.append(" " * indent + f"- {k}:")
            out.extend(emit(v, indent + 4))
        else:
            out.append(" " * indent + f"- {k}: {v}")
    return out

lines = ["- Internals:"] + emit(root, 4)
text = "\n".join(lines) + "\n"

with open(nav_out, "w", encoding="utf-8") as f:
    f.write(text)

print(text, end="")
PY

log_info "Done."
log_info "Wrote markdown stubs under: $INTERNALS_DIR"
log_info "Wrote nav snippet to:      $NAV_OUT"