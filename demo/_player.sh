# ─────────────────────────────────────────────────────────────────────────────
#  Shared plumbing for the demo/*.sh scripts.
#
#  A demo sources this, calls player_start, then alternates say/banner
#  (narration) with run <<'CODE' (typed, executed, output printed in order).
#
#    source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"
#    player_start "title" "subtitle"
#    banner "1 · Something"
#    run <<'CODE'
#    var m = EquityMarket.atmOneYear();
#    CODE
#    finale "closing line"
#
#  This is NOT sourced by any test or build step — it is only ever run by a
#  human. It shells out to jshell against the compiled classes.
#
#  ⚠ Every typed line must be a COMPLETE jshell snippet or obviously
#    unfinished (trailing '(' ',' '{'). jshell appends the missing semicolon,
#    so a fluent chain split across lines becomes two snippets, not one.
#  ⚠ Keep typed lines under ~84 chars or they wrap and the repaint doubles.
#  ⚠ Put a long-running call LAST in its run block: output only streams once
#    every line of the block has been sent.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Box padding counts characters, which bash only does in a UTF-8 locale.
export LC_ALL=C.utf8 LANG=C.utf8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# nablatensor needs JDK 24+ (Class-File API). Deliberately NOT $JAVA_HOME, which
# is often an older JDK. Override with $NABLATENSOR_JDK.
JDK="${NABLATENSOR_JDK:-/opt/zulu26.32.13-ca-jdk26.0.2-linux_x64}"
JSHELL="$JDK/bin/jshell"

TYPE_DELAY=0.012
LINE_PAUSE=0.35
FORCE_CPU=0

for arg in "${DEMO_ARGS[@]:-}"; do
  case "$arg" in
    "")     ;;
    --fast) TYPE_DELAY=0; LINE_PAUSE=0.05 ;;
    --cpu)  FORCE_CPU=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# ── palette ──────────────────────────────────────────────────────────────────
E=$'\033'
RESET="${E}[0m";   BOLD="${E}[1m";      DIM="${E}[38;5;244m"
CYAN="${E}[1;96m"; GREEN="${E}[1;92m";  YELLOW="${E}[1;93m"
WHITE="${E}[1;97m"; MAGENTA="${E}[1;95m"; RED="${E}[1;91m"
KW="${E}[38;5;213m"    # keywords
CMT="${E}[38;5;66m"    # comments
SRC="${E}[38;5;253m"   # everything else

MARK="__NABLATENSOR_EOS__"
W=74

# ── output helpers ───────────────────────────────────────────────────────────
nap() { [[ "$TYPE_DELAY" == "0" ]] || sleep "$1"; }

hr() { local l; printf -v l '%*s' "$W" ''; printf '%s%s%s\n' "$DIM" "${l// /─}" "$RESET"; }

banner() {
  local title="$1" line
  printf -v line '%*s' "$W" ''; line="${line// /─}"
  printf '\n%s╭%s╮%s\n' "$CYAN" "$line" "$RESET"
  printf '%s│%s %s%s%*s%s│%s\n' "$CYAN" "$RESET" "$BOLD" "$title" \
         $(( W - 1 - ${#title} )) '' "$CYAN" "$RESET"
  printf '%s╰%s╯%s\n\n' "$CYAN" "$line" "$RESET"
}

say()  { printf '%s  %s%s\n' "$DIM" "$1" "$RESET"; nap "$LINE_PAUSE"; }
note() { printf '%s  %s%s\n' "$YELLOW" "$1" "$RESET"; }

# Very small Java highlighter — enough to make the typed code look alive.
paint() {
  local line="$1"
  if [[ "$line" =~ ^[[:space:]]*(//|/\*|\*) ]]; then
    printf '%s%s%s' "$CMT" "$line" "$RESET"; return
  fi
  # One sed pass with \b word boundaries: a plain "${line//x/…}" loop wraps the
  # "Products" inside "ExoticProducts" first, leaving no literal "ExoticProducts"
  # for the later pass to match, so only the tail lit up. Longest names first so
  # the alternation prefers "ExoticProducts" over "Products".
  line="$(sed -E "
    s/\\b(ExoticProducts|MonteCarlo|EquityMarket|Calibrator|AadRecorder|SabrHagan|OptionType|Products|Pricing|SDouble|Nabla)\\b/${KW}&${SRC}/g
    s/\\b(var|for|int|void|double|long|new|return)\\b/${KW}&${SRC}/g
  " <<<"$line")"
  printf '%s%s%s' "$SRC" "$line" "$RESET"
}

type_line() {
  local line="$1" prefix="$2" i ch
  if [[ "$TYPE_DELAY" == "0" ]]; then
    printf '%s\n' "$(paint "$line")"; return
  fi
  for (( i = 0; i < ${#line}; i++ )); do
    ch="${line:i:1}"
    printf '%s' "$ch"
    sleep "$TYPE_DELAY"
  done
  printf '\r%s%s\n' "$prefix" "$(paint "$line")"
}

# Prints whatever jshell has already produced, without blocking.
drain() {
  local out
  while IFS= read -r -t 0.02 out <&"${JS[0]}"; do
    if [[ "$out" == *"$MARK"* ]]; then
      out="${out%%"$MARK"*}"
      [[ -n "$out" ]] && printf '%s\n' "$out"
      continue
    fi
    printf '%s\n' "$out"
  done
  return 0
}

# send CODE to jshell, echoing it as if typed, then print whatever comes back
run() {
  local first=1 prefix line
  while IFS= read -r line; do
    if (( first )); then
      prefix="${GREEN}jshell>${RESET} "; first=0
    else
      # drain() can lose a line split across a non-blocking read; only worth the
      # risk when typing is slow enough that mid-block output would otherwise
      # stall. In --fast mode the whole block is sent, then read to the marker.
      [[ "$TYPE_DELAY" == "0" ]] || drain
      prefix="${GREEN}   ...>${RESET} "
    fi
    printf '%s' "$prefix"
    type_line "$line" "$prefix"
    printf '%s\n' "$line" >&"${JS[1]}"
  done
  printf 'System.out.println("%s");\n' "$MARK" >&"${JS[1]}"
  local out
  while IFS= read -r out <&"${JS[0]}"; do
    if [[ "$out" == *"$MARK"* ]]; then
      out="${out%%"$MARK"*}"
      [[ -n "$out" ]] && printf '%s\n' "$out"
      break
    fi
    printf '%s\n' "$out"
  done
  nap "$LINE_PAUSE"
}

# send setup to jshell without showing it
quiet() {
  cat >&"${JS[1]}"
  printf 'System.out.println("%s");\n' "$MARK" >&"${JS[1]}"
  local out
  while IFS= read -r out <&"${JS[0]}"; do
    [[ "$out" == *"$MARK"* ]] && break
  done
}

# like quiet, but returns jshell's output on stdout instead of discarding it
capture() {
  cat >&"${JS[1]}"
  printf 'System.out.println("%s");\n' "$MARK" >&"${JS[1]}"
  local out
  while IFS= read -r out <&"${JS[0]}"; do
    if [[ "$out" == *"$MARK"* ]]; then
      out="${out%%"$MARK"*}"
      [[ -n "$out" ]] && printf '%s\n' "$out"
      break
    fi
    printf '%s\n' "$out"
  done
}

finale() {
  printf '\n'; hr
  local l
  for l in "$@"; do printf '  %s%s%s\n' "$WHITE" "$l" "$RESET"; done
  hr
  printf '  %snablatensor  ·  demo/README.md%s\n\n' "$DIM" "$RESET"
}

# ── launch ───────────────────────────────────────────────────────────────────
player_start() {
  local title="${1:-nablatensor}" subtitle="${2:-}"

  [[ -x "$JSHELL" ]] || { echo "jshell not found at $JSHELL — set NABLATENSOR_JDK" >&2; exit 1; }
  CP="$(find "$ROOT" -path '*/target/classes' -type d | tr '\n' ':')"
  [[ -n "$CP" ]] || { echo "nothing built — run: mvn -o compile" >&2; exit 1; }

  clear
  printf '%s\n' "$DIM"
  cat <<'ART'
    ┌┐┌┌─┐┌┐ ┬  ┌─┐┌┬┐┌─┐┌┐┌┌─┐┌─┐┬─┐
    │││├─┤├┴┐│  ├─┤ │ ├┤ │││└─┐│ │├┬┘
    ┘└┘┴ ┴└─┘┴─┘┴ ┴ ┴ └─┘┘└┘└─┘└─┘┴└─
ART
  printf '%s' "$RESET"
  printf '     %s%s%s\n' "$WHITE" "$title" "$RESET"
  [[ -n "$subtitle" ]] && printf '     %s%s%s\n' "$DIM" "$subtitle" "$RESET"
  printf '\n'
  nap 1.0

  coproc JS { "$JSHELL" -q --class-path "$CP" \
      -R--enable-native-access=ALL-UNNAMED \
      -R--add-modules=jdk.incubator.vector \
      - 2>&1; }

  trap 'printf "/exit\n" >&"${JS[1]}" 2>/dev/null || true' EXIT

  quiet <<'SETUP'
import java.util.*;
import java.util.function.*;
import java.util.Locale;
import com.nablatensor.quant.*;
import com.nablatensor.engine.*;
// Wrap text in a terminal color for the demo's own println output. "sgr" is a
// standard ANSI SGR parameter: 1 = bold, 9x = a bright foreground (2 green,
// 3 yellow, 6 cyan, 7 white), 38;5;N = a color from the 256-color palette.
String color(String sgr, Object s) { return "\033[" + sgr + "m" + s + "\033[0m"; }
String green (Object s) { return color("1;92",     s); }  // headline results
String yellow(Object s) { return color("1;93",     s); }  // timings, strikes
String cyan  (Object s) { return color("1;96",     s); }  // table headers
String white (Object s) { return color("1;97",     s); }  // emphasised numbers
String plain (Object s) { return color("0;97",     s); }  // numbers, unbolded
String grey  (Object s) { return color("38;5;244", s); }  // asides and notes
String silver(Object s) { return color("38;5;250", s); }  // dim row labels
long now() { return System.nanoTime(); }
double msSince(long t) { return (System.nanoTime() - t) / 1e6; }
double time(Runnable r) { long t = now(); r.run(); return msSince(t); }
SETUP

  # Ask nablatensor — not the OS — which adjoint backend is really usable, and
  # in what precision. CUDA/Vulkan/ROCm are fp32; the CPU engines are fp64. The
  # demos that want a GPU read $ENGINE (the fastest one available here); those
  # that need fp64 use "cpu-jit".
  # Override with NABLATENSOR_DEMO_ENGINE=cuda|vulkan|rocm|cpu-jit.
  local ENGINE=""
  if (( FORCE_CPU )); then
    ENGINE="cpu-jit"
  elif [[ -n "${NABLATENSOR_DEMO_ENGINE:-}" ]]; then
    ENGINE="${NABLATENSOR_DEMO_ENGINE}"
  else
    ENGINE="$(capture <<'PROBE'
var _n = AadEngines.discovered().stream().filter(AadEngine::isAvailable).map(AadEngine::name).toList();
System.out.println(_n.contains("cuda") ? "cuda" : _n.contains("vulkan") ? "vulkan" : _n.contains("rocm") ? "rocm" : "cpu-jit");
PROBE
)"
    ENGINE="${ENGINE//[[:space:]]/}"
  fi

  quiet <<SETUP
String ENGINE = "$ENGINE";
boolean GPU = !ENGINE.equals("cpu-jit");
String ENGINE_DESC = AadEngines.discovered().stream().filter(e -> e.name().equals(ENGINE)).map(AadEngine::describe).findFirst().orElse(ENGINE);
System.out.println(grey("engine: " + ENGINE_DESC));
SETUP

  if [[ "$ENGINE" == "cpu-jit" ]]; then
    printf '\n'
    note "No GPU adjoint backend on this machine (or --cpu given) — running on"
    note "the generated-bytecode 'cpu-jit' engine. Every timing below is a CPU"
    note "number; the narration's throughput claims describe the GPU path."
    printf '\n'
    nap 1.2
  fi
}
