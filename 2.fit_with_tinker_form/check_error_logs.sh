#!/bin/bash

# Get the list of error log files
error_logs=$(ls logs/2024-07-13-* | grep err)

# Initialize counters
total_files=0
non_empty_files=0

# Loop through each error log file
while IFS= read -r file; do
    ((total_files++))
    
    # Check if file is non-empty
    if [ -s "$file" ]; then
        ((non_empty_files++))
        echo "Non-empty file: $file"
    fi
    cat $file
done <<< "$error_logs"

# Print results
echo "Total error log files: $total_files"
echo "Non-empty error log files: $non_empty_files"

# Check if all files are empty
if [ $non_empty_files -eq 0 ]; then
    echo "All error log files are empty."
fi
