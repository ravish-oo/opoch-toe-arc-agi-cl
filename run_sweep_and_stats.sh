#!/bin/bash

# Script to run sweep_training_with_miner and append statistics
RESULTS_FILE="docs/human_pair_programmer_docs/results/results_v2.txt"

echo "Running sweep_training_with_miner..."
echo "Output will be saved to: $RESULTS_FILE"
echo ""

# Run the command and capture output, overwriting the file
python -m src.runners.sweep_training_with_miner > "$RESULTS_FILE" 2>&1

# Check if command was successful
if [ $? -ne 0 ]; then
    echo "Warning: Command exited with non-zero status code"
fi

echo ""
echo "Calculating statistics..."

# Calculate statistics from the output file
ERROR_COUNT=$(grep -c "Status: error" "$RESULTS_FILE" 2>/dev/null || echo "0")
INFEASIBLE_COUNT=$(grep -c "Status: infeasible" "$RESULTS_FILE" 2>/dev/null || echo "0")
MISMATCH_TEST_COUNT=$(grep -c "Status: mismatch_test" "$RESULTS_FILE" 2>/dev/null || echo "0")
MISMATCH_TRAIN_COUNT=$(grep -c "Status: mismatch_train" "$RESULTS_FILE" 2>/dev/null || echo "0")
OK_COUNT=$(grep -c "Status: ok" "$RESULTS_FILE" 2>/dev/null || echo "0")
EXCEPTION_COUNT=$(grep -c "Exception" "$RESULTS_FILE" 2>/dev/null || echo "0")

# Remove existing Summary section if it exists (lines starting from "Summary" to end)
# Then append new summary
sed -i.bak '/^Summary$/,$d' "$RESULTS_FILE" 2>/dev/null || sed -i '' '/^Summary$/,$d' "$RESULTS_FILE" 2>/dev/null
rm -f "$RESULTS_FILE.bak" 2>/dev/null

# Append summary section
echo "" >> "$RESULTS_FILE"
echo "======================================================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Summary" >> "$RESULTS_FILE"
echo "Status: error $ERROR_COUNT" >> "$RESULTS_FILE"
echo "Status: infeasible $INFEASIBLE_COUNT" >> "$RESULTS_FILE"
echo "Status: mismatch_test $MISMATCH_TEST_COUNT" >> "$RESULTS_FILE"
echo "Status: mismatch_train $MISMATCH_TRAIN_COUNT" >> "$RESULTS_FILE"
echo "Status: ok $OK_COUNT" >> "$RESULTS_FILE"
echo "Exceptions $EXCEPTION_COUNT" >> "$RESULTS_FILE"

echo "Done! Statistics appended to $RESULTS_FILE"
echo ""
echo "Summary:"
echo "  Status: error $ERROR_COUNT"
echo "  Status: infeasible $INFEASIBLE_COUNT"
echo "  Status: mismatch_test $MISMATCH_TEST_COUNT"
echo "  Status: mismatch_train $MISMATCH_TRAIN_COUNT"
echo "  Status: ok $OK_COUNT"
echo "  Exceptions $EXCEPTION_COUNT"

