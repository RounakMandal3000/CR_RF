#!/usr/bin/env bash
# run_cloudprob_nohup.sh
# Usage:
#   ./run_cloudprob_nohup.sh /path/to/input /path/to/output [workers] [--force]
# Example:
#   ./run_cloudprob_nohup.sh "/c/Users/rouna/OneDrive/Desktop/MTP/ROIs1158_spring_s2_cloudy" "/c/Users/rouna/OneDrive/Desktop/MTP/cloudprob_outputs" 6 --force

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 /path/to/input /path/to/output [workers] [--force]"
  exit 2
fi

INPUT="$1"
OUTPUT="$2"
WORKERS="${3:-}"
FORCE_FLAG=""

# support passing --force as third or fourth arg
if [ "${3:-}" = "--force" ] || [ "${4:-}" = "--force" ]; then
  FORCE_FLAG="--force"
fi

LOG="cloudprob_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="cloudprob.pid"

# Build workers arg
if [ -n "$WORKERS" ] && [ "$WORKERS" != "--force" ]; then
  WORKERS_ARG=( -w "$WORKERS" )
else
  WORKERS_ARG=()
fi

nohup python cloud_probability_extractor.py -i "$INPUT" -o "$OUTPUT" "${WORKERS_ARG[@]}" $FORCE_FLAG > "$LOG" 2>&1 &
PID=$!

echo "$PID" > "$PIDFILE"
echo "Started cloud probability extractor (pid=$PID). Log: $LOG, PID file: $PIDFILE"

echo "To watch the log: tail -f $LOG"

echo "To stop: kill $PID" 
