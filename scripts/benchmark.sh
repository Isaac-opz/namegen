#!/bin/bash
set -e

echo "Starting Benchmark..."
start_time=$(date +%s%N)

# Run 1000 steps and capture the final loss
# We use release mode of course
source $HOME/.cargo/env
cargo run --release > /tmp/benchmark_output 2>&1

end_time=$(date +%s%N)
duration=$(( (end_time - start_time) / 1000000 )) # in milliseconds

echo "--- Benchmark Results ---"
echo "Duration: $duration ms"
echo "Avg step time: $(echo "scale=2; $duration / 1000" | bc) ms"
echo "Final Loss: $(grep "step 1000" /tmp/benchmark_output | tail -n 1 | awk '{print $NF}')"
echo "--------------------------"
rm /tmp/benchmark_output
