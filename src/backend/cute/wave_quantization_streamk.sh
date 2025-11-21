#!/bin/sh

# Determine the basename of the script (without .sh)
SCRIPT_BASENAME="$(basename "$0" .sh)"
OUTFILE="${SCRIPT_BASENAME}.txt"

# Number of SMs = 36 → BN runs from 1 to (36*4+4) = 148
MAX_BN=$((36*4 + 4))

echo "Writing output to ${OUTFILE}"
echo "==== StreamK sweep started at $(date) ====" >> "${OUTFILE}"

for BN in $(seq 1 "$MAX_BN"); do
    N=$((128 * BN))
    echo "Running BN=${BN}, n=${N}"
    echo "--- BN=${BN}, n=${N} ---" >> "${OUTFILE}"

    ./sm80_streamk --m=128 --n="${N}" --k=32 >> "${OUTFILE}" 2>&1
done

echo "==== StreamK sweep finished at $(date) ====" >> "${OUTFILE}"
