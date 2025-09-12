#!/bin/zsh
# This script runs a specified Python script multiple times with a given input file and shows timing statistics.

# --- User Input ---

# Prompt for the Python script name (default: summarizeDocument.py)
echo -n "Enter the Python script to run [default: summarizeDocument.py]: "
read python_script
python_script=${python_script:-"summarizeDocument.py"}

# Prompt for the input file name (required)
echo -n "Enter the input file path: "
read input_file
while [[ ! -f "$input_file" ]]; do
  echo "File not found. Please enter a valid file path."
  echo -n "Enter the input file path: "
  read input_file
done

# Prompt for the number of iterations
echo -n "Enter the number of iterations (e.g., 20): "
read num_iterations
while ! [[ "$num_iterations" =~ ^[0-9]+$ ]] || [[ "$num_iterations" -lt 1 ]]; do
  echo "Invalid input. Please enter a positive integer."
  echo -n "Enter the number of iterations: "
  read num_iterations
done

# --- Script Logic ---

# Create a temporary file to store the output of each run
TMP_FILE=$(mktemp)

echo -e "\nStarting tests with file: $input_file"
echo "Running $python_script for $num_iterations iterations..."

for ((i=1; i<=num_iterations; i++)); do
    echo "--- Running test $i of $num_iterations ---"
    { /usr/bin/time -p python3.10 "$python_script" "$input_file"; } 2>> "$TMP_FILE"
done

# --- Statistics ---

echo -e "\n--- Summary Statistics ---"
echo "Real times (in seconds):"
awk '/real/ {print $2}' "$TMP_FILE"

awk '
    /real/ {
        total += $2;
        count++;
        if (NR == 1 || $2 > max) max = $2;
        if (NR == 1 || $2 < min) min = $2;
    }
    END {
        if (count > 0) {
            print "\nTotal runs: " count
            printf "Average real time: %.3f s\n", total/count
            printf "Minimum real time: %.3f s\n", min
            printf "Maximum real time: %.3f s\n", max
        } else {
            print "No 'real' time values found. Please check the script and output."
        }
    }' "$TMP_FILE"

rm "$TMP_FILE"
