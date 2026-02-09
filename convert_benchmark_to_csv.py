#!/usr/bin/env python3
"""
Convert benchmark_results.json (per-puzzle results) to results.csv (aggregated format).

This script reads a JSONL file containing per-puzzle evaluation results and converts it
to the aggregated CSV format required by the leaderboard.

Usage:
    python convert_benchmark_to_csv.py [input_file] [output_file]
    
    If input_file is not provided, defaults to 'benchmark_results.json'
    If output_file is not provided, defaults to 'leaderboard/results.csv'
"""

import json
import csv
import argparse
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional


def infer_type(record: Dict) -> str:
    """Infer model type (Proprietary/Open source) from Provider or Model name."""
    provider = str(record.get("Provider", "")).lower()
    model = str(record.get("Model", "")).lower()
    
    # Open source indicators
    open_source_keywords = [
        "llama", "mistral", "qwen", "phi", "gemma", "falcon", "mpt", 
        "vicuna", "alpaca", "wizard", "openchat", "neural-chat",
        "browser-use", "browseruse", "open source", "opensource", "bu-1"
    ]
    
    # Check if any open source keyword appears
    for keyword in open_source_keywords:
        if keyword in provider or keyword in model:
            return "Open source"
    
    # Default to Proprietary if not found
    return "Proprietary"


def normalize_provider(provider: str) -> str:
    """Normalize provider name to standard format."""
    provider_lower = provider.lower()
    
    if provider_lower in ['openai', 'gpt']:
        return 'OpenAI'
    elif provider_lower in ['anthropic', 'claude']:
        return 'Anthropic'
    elif provider_lower in ['google', 'gemini']:
        return 'Google'
    elif provider_lower in ['browser-use', 'browseruse']:
        return 'browser-use'
    else:
        # Capitalize first letter of each word
        return provider.title()


def convert_benchmark_results_json(
    input_file: pathlib.Path,
    output_file: Optional[pathlib.Path] = None
) -> None:
    """
    Convert benchmark_results.json format (per-puzzle results) to aggregated CSV format.
    
    Args:
        input_file: Path to benchmark_results.json file (JSONL format)
        output_file: Path to output CSV file (optional)
    """
    # Read the file - it's a JSONL file (one JSON object per line)
    puzzle_results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    puzzle_results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    if not puzzle_results:
        raise ValueError("No valid puzzle results found in file")
    
    print(f"Loaded {len(puzzle_results)} puzzle results")
    
    # Group results by model/provider/agent_framework combination
    # This allows handling multiple models in one file
    grouped_results = defaultdict(list)
    
    for result in puzzle_results:
        # Extract metadata
        model = result.get('model') or 'Unknown Model'
        provider = result.get('provider') or 'Unknown'
        agent_framework = result.get('agent_framework') or result.get('agentFramework') or 'browser-use'
        
        # Normalize provider
        provider = normalize_provider(provider)
        
        # Create a key for grouping
        key = (model, provider, agent_framework)
        grouped_results[key].append(result)
    
    print(f"Found {len(grouped_results)} unique model/provider/framework combinations")
    
    # Process each group
    all_records = []
    
    for (model, provider, agent_framework), results in grouped_results.items():
        print(f"\nProcessing: {model} ({provider}, {agent_framework}) - {len(results)} results")
        
        # Group by puzzle_type
        puzzle_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_correct = 0
        total_attempts = len(results)
        total_duration = 0.0
        total_cost = 0.0
        cost_count = 0
        
        for result in results:
            puzzle_type = result.get('puzzle_type', 'Unknown')
            
            puzzle_type_stats[puzzle_type]['total'] += 1
            if result.get('correct', False):
                puzzle_type_stats[puzzle_type]['correct'] += 1
                total_correct += 1
            
            # Aggregate duration
            elapsed_time = result.get('elapsed_time')
            if elapsed_time is not None:
                try:
                    total_duration += float(elapsed_time)
                except (ValueError, TypeError):
                    pass
            
            # Aggregate cost
            cost = result.get('cost')
            if cost is not None:
                try:
                    total_cost += float(cost)
                    cost_count += 1
                except (ValueError, TypeError):
                    pass
        
        # Calculate overall pass rate
        overall_pass_rate = total_correct / total_attempts if total_attempts > 0 else 0.0
        
        # Calculate average duration
        avg_duration = total_duration / total_attempts if total_attempts > 0 else None
        
        # Calculate average cost
        avg_cost = total_cost / cost_count if cost_count > 0 else None
        
        # Build aggregated record
        record = {
            "Model": model,
            "Provider": provider,
            "Agent Framework": agent_framework,
            "Overall Pass Rate": overall_pass_rate,
            "Avg Duration (s)": avg_duration if avg_duration is not None else "",
            "Avg Cost ($)": avg_cost if avg_cost is not None else "",
        }
        
        # Add per-type pass rates
        for puzzle_type, stats in puzzle_type_stats.items():
            pass_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            record[puzzle_type] = pass_rate
        
        # Infer Type
        record["Type"] = infer_type(record)
        
        all_records.append(record)
        
        print(f"  Overall Pass Rate: {overall_pass_rate:.2%}")
        print(f"  Puzzle types: {list(puzzle_type_stats.keys())}")
    
    # Determine all puzzle type columns (union of all puzzle types across all records)
    all_puzzle_types = set()
    for record in all_records:
        all_puzzle_types.update(
            k for k in record.keys() 
            if k not in ["Model", "Provider", "Agent Framework", "Type", 
                        "Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)"]
        )
    
    # Sort puzzle types for consistent column order
    all_puzzle_types = sorted(all_puzzle_types)
    
    # Define column order: metadata → metrics → puzzle types
    fixed_metadata = ["Model", "Provider", "Agent Framework", "Type"]
    fixed_metrics = ["Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)"]
    header = fixed_metadata + fixed_metrics + all_puzzle_types
    
    # Ensure all records have all columns (fill missing puzzle types with empty string)
    for record in all_records:
        for puzzle_type in all_puzzle_types:
            if puzzle_type not in record:
                record[puzzle_type] = ""
    
    # Write CSV
    if output_file is None:
        output_file = pathlib.Path("leaderboard/results.csv")
    else:
        output_file = pathlib.Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for record in all_records:
            writer.writerow(record)
    
    print(f"\n✅ Successfully converted to {output_file}")
    print(f"   Created {len(all_records)} aggregated record(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark_results.json to results.csv format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert using defaults (benchmark_results.json -> leaderboard/results.csv)
  python convert_benchmark_to_csv.py
  
  # Specify input and output files
  python convert_benchmark_to_csv.py input.json output.csv
  
  # Convert and append to existing results.csv
  python convert_benchmark_to_csv.py results.json leaderboard/results.csv
        """
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='benchmark_results.json',
        help='Input JSONL file (default: benchmark_results.json)'
    )
    parser.add_argument(
        'output_file',
        nargs='?',
        default=None,
        help='Output CSV file (default: leaderboard/results.csv)'
    )
    
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    try:
        convert_benchmark_results_json(input_path, args.output_file)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

