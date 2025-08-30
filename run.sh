#!/usr/bin/env bash

# Replace "myfile.txt" with your file
while IFS= read -r line; do
    # Skip empty lines (optional)
    [ -z "$line" ] && continue
    
    echo "Running: python main.py --repo $line --docker benchmark-image --cycles 75 --script"
    if ! python main.py --repo "$line" --docker benchmark-image --cycles 75 --script 2>>error.txt; then
        echo "❌ Failed on repo: $line" >> error.txt
        echo "----------------------------------------" >> error.txt
    fi
done < myfile.txt