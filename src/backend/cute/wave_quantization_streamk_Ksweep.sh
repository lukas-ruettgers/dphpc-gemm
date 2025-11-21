#!/bin/sh

# Determine the basename of the script (without .sh)
SCRIPT_BASENAME="$(basename "$0" .sh)"
OUTFILE="${SCRIPT_BASENAME}.txt"

# Number of SMs = 36 → BN runs from 1 to (36*4+4) = 148
MAX_BN=$((36*4 + 4))

echo "Writing output to ${OUTFILE}"
echo "==== StreamK sweep started at $(date) ====" >> "${OUTFILE}"

for BK in $(seq 1 "$MAX_BN"); do
    K=$((32 * BK))
    echo "Running BK=${BK}, k=${K}"
    echo "--- BK=${BK}, n=${K} ---" >> "${OUTFILE}"

    ./sm80_streamk --m=128 --n=128 --k="${K}" >> "${OUTFILE}" 2>&1
done

echo "==== StreamK sweep finished at $(date) ====" >> "${OUTFILE}"
