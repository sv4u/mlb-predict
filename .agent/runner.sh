#!/usr/bin/env bash
# .agent/runner.sh — Universal CLI runner for .Rmd agent files
#
# Parses an .Rmd file for directives and executes them in order.
# Phase 1: check all /require: dependencies (fail-fast)
# Phase 2: execute directives sequentially (top-to-bottom)
#
# Usage:
#   .agent/runner.sh <agent-file.Rmd>
#   .agent/runner.sh --dry-run <agent-file.Rmd>
#   .agent/runner.sh --line <N> <agent-file.Rmd>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

DRY_RUN=0
TARGET_LINE=0
WRITE_BACK=0
NON_INTERACTIVE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --line) TARGET_LINE="$2"; shift 2 ;;
        --write-back) WRITE_BACK=1; shift ;;
        --non-interactive) NON_INTERACTIVE=1; shift ;;
        *) break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--dry-run] [--line N] [--non-interactive] <agent-file.Rmd>" >&2
    exit 1
fi

AGENT_FILE="$1"
if [[ ! -f "$AGENT_FILE" ]]; then
    echo "Error: agent file not found: $AGENT_FILE" >&2
    exit 1
fi

ERROR_MODE="halt"
declare -A CAPTURED_VARS

resolve_runner() {
    local ext="${1##*.}"
    local runner_file
    runner_file=$(grep -E "^\s+\.${ext}:" "$CONFIG" 2>/dev/null | head -1 | awk '{print $2}')
    if [[ -n "$runner_file" ]]; then
        echo "$SCRIPT_DIR/runners/$runner_file"
    else
        echo ""
    fi
}

check_requirement() {
    local req="$1"
    local tool version_constraint
    tool=$(echo "$req" | awk '{print $1}')
    version_constraint=$(echo "$req" | awk '{$1=""; print}' | xargs)

    if ! command -v "$tool" > /dev/null 2>&1; then
        echo "FAIL: $tool not found in PATH" >&2
        return 1
    fi

    if [[ -n "$version_constraint" ]]; then
        echo "  OK: $tool (version check: $version_constraint — manual verification recommended)" >&2
    else
        echo "  OK: $tool" >&2
    fi
    return 0
}

run_script() {
    local script_path="$1"; shift
    local full_path

    if [[ "$script_path" == /* ]]; then
        full_path="$script_path"
    else
        full_path="$REPO_ROOT/$script_path"
    fi

    if [[ ! -f "$full_path" ]]; then
        echo "Error: script not found: $full_path" >&2
        return 1
    fi

    local runner
    runner=$(resolve_runner "$full_path")

    if [[ -z "$runner" || ! -f "$runner" ]]; then
        echo "Error: no runner found for extension of $script_path" >&2
        return 1
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] $runner $full_path $*"
        return 0
    fi

    echo "--- Running: $script_path $* ---" >&2
    "$runner" "$full_path" "$@"
}

handle_error() {
    local exit_code=$1
    local directive="$2"

    if [[ $exit_code -eq 0 ]]; then
        return 0
    fi

    case "$ERROR_MODE" in
        halt)
            echo "HALT: directive failed (exit $exit_code): $directive" >&2
            exit $exit_code
            ;;
        continue)
            echo "CONTINUE: directive failed (exit $exit_code), proceeding: $directive" >&2
            return 0
            ;;
        *)
            echo "HALT: unknown error mode '$ERROR_MODE', failing" >&2
            exit $exit_code
            ;;
    esac
}

parse_retry_max() {
    local mode="$1"
    local n
    n=$(echo "$mode" | awk '{print $2}')
    echo "${n:-3}"
}

run_with_retry() {
    local script="$1"
    local args="$2"
    local directive="$3"
    local max_retries
    max_retries=$(parse_retry_max "$ERROR_MODE")
    local attempt=0
    local delay=1
    local rc=0

    while [[ $attempt -le $max_retries ]]; do
        rc=0
        if [[ $attempt -gt 0 ]]; then
            echo "RETRY: attempt $attempt/$max_retries (backoff ${delay}s): $directive" >&2
            sleep "$delay"
            delay=$((delay * 2))
        fi
        if [[ -n "$args" ]]; then
            run_script "$script" $args || rc=$?
        else
            run_script "$script" || rc=$?
        fi
        if [[ $rc -eq 0 ]]; then
            return 0
        fi
        attempt=$((attempt + 1))
    done

    echo "RETRY EXHAUSTED: failed after $max_retries retries (exit $rc): $directive" >&2
    exit $rc
}

echo "=== Phase 1: Dependency Check ===" >&2
REQUIRE_FAIL=0
while IFS= read -r line; do
    req="${line#/require: }"
    if ! check_requirement "$req"; then
        REQUIRE_FAIL=1
    fi
done < <(grep -E '^/require:' "$AGENT_FILE" 2>/dev/null || true)

if [[ $REQUIRE_FAIL -ne 0 ]]; then
    echo "FATAL: one or more dependencies are missing" >&2
    exit 1
fi
echo "=== All dependencies satisfied ===" >&2
echo "" >&2

echo "=== Phase 2: Sequential Execution ===" >&2
LINE_NUM=0

# Conditional block state — stack-based to support nesting.
# Each entry is "1" (executing) or "0" (skipping).
# SKIP_DEPTH tracks how many nested /if: levels deep we are inside a
# skipped branch so that inner /if:../endif pairs are consumed without
# altering the outer state.
IF_STACK=()
SKIP_DEPTH=0

should_execute() {
    if [[ $SKIP_DEPTH -gt 0 ]]; then
        return 1
    fi
    for state in "${IF_STACK[@]}"; do
        if [[ "$state" == "0" ]]; then
            return 1
        fi
    done
    return 0
}

while IFS= read -r line || [[ -n "$line" ]]; do
    LINE_NUM=$((LINE_NUM + 1))

    if [[ $TARGET_LINE -gt 0 && $LINE_NUM -ne $TARGET_LINE ]]; then
        continue
    fi

    # --- Conditional directives are always processed (even inside skipped blocks) ---

    # /if: <shell-expression>
    if [[ "$line" =~ ^/if:\ (.+) ]]; then
        local_expr="${BASH_REMATCH[1]}"
        if [[ $SKIP_DEPTH -gt 0 ]]; then
            SKIP_DEPTH=$((SKIP_DEPTH + 1))
            echo "[if] (nested skip, depth=$SKIP_DEPTH) $local_expr" >&2
        else
            if should_execute; then
                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "[DRY RUN] if: $local_expr (assumed true)"
                    IF_STACK+=("1")
                elif eval "$local_expr" 2>/dev/null; then
                    echo "[if] TRUE: $local_expr" >&2
                    IF_STACK+=("1")
                else
                    echo "[if] FALSE: $local_expr" >&2
                    IF_STACK+=("0")
                fi
            else
                SKIP_DEPTH=$((SKIP_DEPTH + 1))
                echo "[if] (nested skip, depth=$SKIP_DEPTH) $local_expr" >&2
            fi
        fi
        continue
    fi

    # /else
    if [[ "$line" =~ ^/else ]]; then
        if [[ $SKIP_DEPTH -gt 0 ]]; then
            echo "[else] (nested skip, depth=$SKIP_DEPTH)" >&2
        elif [[ ${#IF_STACK[@]} -eq 0 ]]; then
            echo "ERROR (line $LINE_NUM): /else without matching /if:" >&2
            exit 1
        else
            local_idx=$(( ${#IF_STACK[@]} - 1 ))
            if [[ "${IF_STACK[$local_idx]}" == "1" ]]; then
                IF_STACK[$local_idx]="0"
                echo "[else] flipped to SKIP" >&2
            else
                IF_STACK[$local_idx]="1"
                echo "[else] flipped to EXECUTE" >&2
            fi
        fi
        continue
    fi

    # /endif
    if [[ "$line" =~ ^/endif ]]; then
        if [[ $SKIP_DEPTH -gt 0 ]]; then
            SKIP_DEPTH=$((SKIP_DEPTH - 1))
            echo "[endif] (nested skip, depth=$SKIP_DEPTH)" >&2
        elif [[ ${#IF_STACK[@]} -eq 0 ]]; then
            echo "ERROR (line $LINE_NUM): /endif without matching /if:" >&2
            exit 1
        else
            unset 'IF_STACK[${#IF_STACK[@]}-1]'
            echo "[endif]" >&2
        fi
        continue
    fi

    # --- All other directives respect the conditional state ---

    if ! should_execute; then
        continue
    fi

    # /env: KEY=VALUE
    if [[ "$line" =~ ^/env:\ (.+) ]]; then
        local_env="${BASH_REMATCH[1]}"
        eval "export $local_env"
        echo "[env] $local_env" >&2
        continue
    fi

    # /on-error: <mode>
    if [[ "$line" =~ ^/on-error:\ (.+) ]]; then
        ERROR_MODE="${BASH_REMATCH[1]}"
        echo "[on-error] mode=$ERROR_MODE" >&2
        continue
    fi

    # /assert: <expression>
    if [[ "$line" =~ ^/assert:\ (.+) ]]; then
        local_expr="${BASH_REMATCH[1]}"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[DRY RUN] assert: $local_expr"
            continue
        fi
        echo "[assert] $local_expr" >&2
        if ! eval "$local_expr"; then
            echo "ASSERT FAILED (line $LINE_NUM): $local_expr" >&2
            exit 1
        fi
        continue
    fi

    # /run: <script> [args...]
    if [[ "$line" =~ ^/run:\ (.+) ]]; then
        local_cmd="${BASH_REMATCH[1]}"
        local_script=$(echo "$local_cmd" | awk '{print $1}')
        local_args=$(echo "$local_cmd" | awk '{$1=""; print}' | xargs)

        if [[ "$ERROR_MODE" == retry* ]]; then
            run_with_retry "$local_script" "$local_args" "/run: $local_cmd"
        else
            local_rc=0
            if [[ -n "$local_args" ]]; then
                run_script "$local_script" $local_args || local_rc=$?
            else
                run_script "$local_script" || local_rc=$?
            fi
            handle_error $local_rc "/run: $local_cmd" || true
        fi
        continue
    fi

    # /ai: directives are skipped in CLI mode
    if [[ "$line" =~ ^/ai:\ (.+) ]]; then
        echo "[skip] /ai: directive (CLI mode — use Cursor for AI directives)" >&2
        continue
    fi

    # /prompt: directives
    if [[ "$line" =~ ^/prompt:\ ([A-Z_]+)(=[^ ]*)?\ (.+) ]]; then
        local_var_name="${BASH_REMATCH[1]}"
        local_default="${BASH_REMATCH[2]#=}"
        local_message="${BASH_REMATCH[3]}"

        if [[ $NON_INTERACTIVE -eq 1 ]]; then
            if [[ -n "$local_default" ]]; then
                export "$local_var_name=$local_default"
                echo "[prompt] $local_var_name=$local_default (non-interactive default)" >&2
            else
                echo "FATAL: /prompt: $local_var_name requires input but running in --non-interactive mode" >&2
                exit 1
            fi
        else
            echo "$local_message" >&2
            read -r -p "> " user_input
            export "$local_var_name=${user_input:-$local_default}"
            echo "[prompt] $local_var_name=${!local_var_name}" >&2
        fi
        continue
    fi

done < "$AGENT_FILE"

if [[ ${#IF_STACK[@]} -gt 0 ]]; then
    echo "ERROR: ${#IF_STACK[@]} unclosed /if: block(s) at end of file" >&2
    exit 1
fi

echo "" >&2
echo "=== Pipeline complete ===" >&2
