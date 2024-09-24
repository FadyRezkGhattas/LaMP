#!/bin/bash

function launch_with_gpu {
  local pre_launch_commands="cd /disk/scratch1/fady && source /opt/conda/bin/activate ./liouna && cd Liouna"
  local cmd="$1"
  local gpu_id="$2"
  tmux new-window -n "GPU $gpu_id" \; send-keys "$pre_launch_commands && export CUDA_VISIBLE_DEVICES=$gpu_id && $cmd" C-m
}

# Check for enough arguments
if [[ $# -lt 3 ]]; then
  echo "Error: Please provide the start GPU index, end GPU index, and the commands file path"
  exit 1
fi

# Get start and end GPU indices
start_index=$1
end_index=$2

# Get commands file path
commands_file=$3

# Check if commands file exists
if [[ ! -f "$commands_file" ]]; then
  echo "Error: Commands file not found: $commands_file"
  exit 1
fi

# Check if start and end indices are valid
if [[ $start_index -gt $end_index ]]; then
  echo "Error: Start index must be less than or equal to end index"
  exit 1
fi

# Check if enough GPUs are available
num_gpus_available=$((end_index - start_index + 1))
if [[ $num_gpus_available -lt $(wc -l < "$commands_file") ]]; then
  echo "Error: Not enough GPUs available for the given number of commands"
  exit 1
fi

# Read commands line by line using a loop
count=0
while IFS= read -r line; do
  # Check if we reached the desired number of commands
  if [[ $count -ge $num_gpus_available ]]; then
    break
  fi
  gpu_id=$((start_index + count))
  launch_with_gpu "$line" "$gpu_id"
  ((count++))
done < "$commands_file"
