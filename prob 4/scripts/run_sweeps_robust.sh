#!/usr/bin/env bash
# 稳健重跑三个参数重扫：每套独立进程 + 崩溃自动重试（6.4 同源非确定性崩溃）。
set -u
cd "/c/Users/ASUS/Desktop/math/prob 4" || exit 1
PY="/c/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe"

run_until_done() {
    local name="$1"; shift
    local out="$1"; shift
    local attempt
    for attempt in 1 2 3 4 5 6; do
        echo "=== [$name] attempt $attempt ==="
        "$PY" "$@" > "$out" 2>&1
        if grep -q "^suite=" "$out"; then
            echo "=== [$name] DONE (attempt $attempt) ==="
            return 0
        fi
        echo "=== [$name] crashed, retrying ($(tail -1 "$out" | head -c 120)) ==="
    done
    echo "=== [$name] FAILED after 6 attempts ==="
    return 1
}

run_until_done "bisect_pruning" "outputs/sweep_bisect_pruning.txt" \
    scripts/ab_test.py --suite bisect_probe --cases 400 --start-seed 14000
run_until_done "verify_pruning" "outputs/sweep_verify_pruning.txt" \
    scripts/ab_test.py --suite verify_fine --cases 400 --start-seed 14000
run_until_done "bisect_dir1" "outputs/sweep_bisect_dir1.txt" \
    scripts/ab_test.py --suite bisect_probe --cases 300 --start-seed 20000 --directional-prob 1.0

echo "ALL SWEEPS DONE"
