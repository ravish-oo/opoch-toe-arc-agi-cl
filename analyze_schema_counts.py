#!/usr/bin/env python3
"""
Analyze schema instance counts from results file and bucket them.
"""

import re
import sys
from collections import defaultdict

def bucket_schema_count(count):
    """Assign a schema count to a bucket."""
    if count < 50:
        return "0-50"
    elif count < 100:
        return "50-100"
    elif count < 200:
        return "100-200"
    elif count < 400:
        return "200-400"
    else:
        return "400+"

def analyze_results_file(filepath):
    """Parse results file and count schema instances by bucket."""
    bucket_counts = defaultdict(int)
    total_tasks = 0
    pattern = re.compile(r'Mined (\d+) schema instances')
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    count = int(match.group(1))
                    bucket = bucket_schema_count(count)
                    bucket_counts[bucket] += 1
                    total_tasks += 1
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return bucket_counts, total_tasks

def main():
    filepath = "docs/human_pair_programmer_docs/results/results_v2.txt"
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    bucket_counts, total_tasks = analyze_results_file(filepath)
    
    # Print results
    print(f"Schema Instance Count Analysis")
    print(f"File: {filepath}")
    print(f"Total tasks: {total_tasks}")
    print()
    print("Bucket distribution:")
    
    # Print in order
    buckets = ["0-50", "50-100", "100-200", "200-400", "400+"]
    for bucket in buckets:
        count = bucket_counts[bucket]
        percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
        print(f"  {bucket:10s}: {count:4d} tasks ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()

