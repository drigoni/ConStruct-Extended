import re
import numpy as np
import glob
import os
from collections import defaultdict

# Original patterns for molecular metrics
patterns = {
    "FCD": r"\bFCD\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Unique": r"\bUnique\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Novel": r"\bNovel\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Valid": r"\bValid\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Disconnected": r"\bDisconnected\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "VUN": r"\bV\.U\.N\.\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Satisfied": r"\b(?:Property|Planarity|Ring count|Cycle rank) satisfied\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
}

# Labels as shown in the table-style summaries
TABLE_LABELS = {
    "FCD": r"^\|\s*FCD\s*\|",
    "Unique": r"^\|\s*Unique\s*\(\%\)\s*\|",
    "Novel": r"^\|\s*Novel\s*\(\%\)\s*\|",
    "Valid": r"^\|\s*Valid\s*\(\%\)\s*\|",
    "Disconnected": r"^\|\s*Disconnected\s*\(\%\)\s*\|",
    "VUN": r"^\|\s*V\.U\.N\.\s*\(\%\)\s*\|",
    "Satisfied": r"^\|\s*(?:Property|Planarity|Ring count|Cycle rank) satisfied\s*\(\%\)\s*\|",
}

# New patterns for core metrics table
CORE_METRICS_PATTERNS = {
    "molecules_generated": r"^\|\s*Molecules generated\s*\(N\)\s*\|",
    "FCD": r"^\|\s*FCD\s*\|",
    "Unique": r"^\|\s*Unique\s*\(\%\)\s*\|",
    "Novel": r"^\|\s*Novel\s*\(\%\)\s*\|",
    "Valid": r"^\|\s*Valid\s*\(\%\)\s*\|",
    "Disconnected": r"^\|\s*Disconnected\s*\(\%\)\s*\|",
    "Property_satisfied": r"^\|\s*Property satisfied\s*\(\%\)\s*\|",
    "VUN": r"^\|\s*V\.U\.N\.\s*\(\%\)\s*\|",
}

# New patterns for structural constraint satisfaction tables
STRUCTURAL_PATTERNS = {
    # Ring count patterns (renamed from cycle rank)
    "ring_count_0": r"^\|\s*Ring count 0\s*\(\%\)\s*\|",
    "ring_count_1": r"^\|\s*Ring count 1\s*\(\%\)\s*\|",
    "ring_count_2": r"^\|\s*Ring count 2\s*\(\%\)\s*\|",
    "ring_count_3": r"^\|\s*Ring count 3\s*\(\%\)\s*\|",
    "ring_count_4": r"^\|\s*Ring count 4\s*\(\%\)\s*\|",
    "ring_count_5": r"^\|\s*Ring count 5\s*\(\%\)\s*\|",
    "ring_count_6": r"^\|\s*Ring count 6\s*\(\%\)\s*\|",
    "ring_count_7": r"^\|\s*Ring count 7\s*\(\%\)\s*\|",
    "ring_count_8": r"^\|\s*Ring count 8\s*\(\%\)\s*\|",
    "ring_count_9": r"^\|\s*Ring count 9\s*\(\%\)\s*\|",
    "ring_count_9plus": r"^\|\s*Ring count 9\+\s*\(\%\)\s*\|",
    "ring_count_12": r"^\|\s*Ring count 12\s*\(\%\)\s*\|",
    "ring_count_40": r"^\|\s*Ring count 40\s*\(\%\)\s*\|",
    "acyclic": r"^\|\s*Acyclic\s*\(max len 0\)\s*\(\%\)\s*\|",
    
    # Ring length patterns (using actual output format: Cycle length X (max) (%))
    "ring_length_3": r"^\|\s*(?:Cycle|Ring) length 3\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_4": r"^\|\s*(?:Cycle|Ring) length 4\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_5": r"^\|\s*(?:Cycle|Ring) length 5\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_6": r"^\|\s*(?:Cycle|Ring) length 6\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_7": r"^\|\s*(?:Cycle|Ring) length 7\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_8": r"^\|\s*(?:Cycle|Ring) length 8\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_9": r"^\|\s*(?:Cycle|Ring) length 9\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_10": r"^\|\s*(?:Cycle|Ring) length 10\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_11": r"^\|\s*(?:Cycle|Ring) length 11\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_12": r"^\|\s*(?:Cycle|Ring) length 12\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_12plus": r"^\|\s*(?:Cycle|Ring) length >12\s*\(max\)\s*\(\%\)\s*\|",
    
    # Violation patterns (greater than thresholds) - using actual output format
    "ring_count_gt0": r"^\|\s*Ring count >0\s*\(\%\)\s*\|",
    "ring_count_gt1": r"^\|\s*Ring count >1\s*\(\%\)\s*\|",
    "ring_count_gt2": r"^\|\s*Ring count >2\s*\(\%\)\s*\|",
    "ring_count_gt3": r"^\|\s*Ring count >3\s*\(\%\)\s*\|",
    "ring_count_gt4": r"^\|\s*Ring count >4\s*\(\%\)\s*\|",
    "ring_count_gt5": r"^\|\s*Ring count >5\s*\(\%\)\s*\|",
    "ring_count_gt6": r"^\|\s*Ring count >6\s*\(\%\)\s*\|",
    "ring_count_gt7": r"^\|\s*Ring count >7\s*\(\%\)\s*\|",
    "ring_count_gt9": r"^\|\s*Ring count >9\s*\(\%\)\s*\|",
    "ring_count_gt12": r"^\|\s*Ring count >12\s*\(\%\)\s*\|",
    "ring_count_gt40": r"^\|\s*Ring count >40\s*\(\%\)\s*\|",
    "ring_length_gt3": r"^\|\s*(?:Cycle|Ring) length >3\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_gt4": r"^\|\s*(?:Cycle|Ring) length >4\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_gt5": r"^\|\s*(?:Cycle|Ring) length >5\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_gt6": r"^\|\s*(?:Cycle|Ring) length >6\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_gt7": r"^\|\s*(?:Cycle|Ring) length >7\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_gt8": r"^\|\s*(?:Cycle|Ring) length >8\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_gt9": r"^\|\s*(?:Cycle|Ring) length >9\s*\(max\)\s*\(\%\)\s*\|",
    "ring_length_gt12": r"^\|\s*(?:Cycle|Ring) length >12\s*\(max\)\s*\(\%\)\s*\|",
}

def _parse_table_style(text):
    results = {}
    lines = text.splitlines()
    for line in lines:
        for key, label_regex in TABLE_LABELS.items():
            if re.search(label_regex, line):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    value_str = parts[2]
                    value_str = value_str.rstrip('%').strip()
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value_str)
                    if m:
                        try:
                            if key not in results:
                                results[key] = []
                            results[key].append(float(m.group(1)))
                        except ValueError:
                            pass
    return results

def _parse_core_metrics(text):
    """Parse the CORE METRICS section - capture all 5 seeds per file"""
    results = {}
    lines = text.splitlines()
    
    in_core_metrics = False
    in_table = False
    current_section_results = {}
    
    for line in lines:
        if "🎯 CORE METRICS:" in line:
            # If we were in a previous section, save its results
            if in_core_metrics and current_section_results:
                for key, value in current_section_results.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
                current_section_results = {}
            
            in_core_metrics = True
            in_table = False
            continue
            
        if in_core_metrics and "| Metric" in line:
            in_table = True
            continue
            
        if in_table and line.strip().startswith("+---"):
            continue
            
        if in_table and line.strip() == "":
            in_table = False
            continue
            
        if in_table and "|" in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                metric_name = parts[1].strip()
                value_str = parts[2].strip()
                
                # Extract numeric value
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value_str)
                if m:
                    try:
                        value = float(m.group(1))
                        
                        # Map metric names to our standardized keys
                        if "Molecules generated" in metric_name:
                            key = "molecules_generated"
                        elif "FCD" in metric_name:
                            key = "FCD"
                        elif "Unique" in metric_name:
                            key = "Unique"
                        elif "Novel" in metric_name:
                            key = "Novel"
                        elif "Valid" in metric_name:
                            key = "Valid"
                        elif "Disconnected" in metric_name:
                            key = "Disconnected"
                        elif "Property satisfied" in metric_name:
                            key = "Property_satisfied"
                        elif "V.U.N." in metric_name:
                            key = "VUN"
                        else:
                            continue
                            
                        current_section_results[key] = value
                    except ValueError:
                        pass
        
        # Check for end of core metrics section
        if in_core_metrics and ("🏗️ STRUCTURAL CONSTRAINT SATISFACTION:" in line or "⏱️ TIMING METRICS:" in line):
            # Save the current section results
            if current_section_results:
                for key, value in current_section_results.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
                current_section_results = {}
            in_core_metrics = False
            in_table = False
    
    # Don't forget the last section
    if in_core_metrics and current_section_results:
        for key, value in current_section_results.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    
    return results

def _parse_structural_table(text):
    """Parse structural constraint satisfaction tables"""
    results = {}
    lines = text.splitlines()
    
    # Find all structural constraint satisfaction sections
    structural_sections = []
    current_section = []
    in_structural_section = False
    in_table = False
    
    for line in lines:
        if "🏗️ STRUCTURAL CONSTRAINT SATISFACTION:" in line:
            # If we were in a previous section, save it
            if current_section:
                structural_sections.append(current_section)
            # Start new section
            current_section = [line]
            in_structural_section = True
            continue
            
        if in_structural_section:
            current_section.append(line)
            
            if "| Metric" in line:
                in_table = True
                continue
                
            if in_table and line.strip().startswith("+---"):
                continue
                
            if in_table and line.strip() == "":
                in_table = False
                in_structural_section = False
                # Don't break here, continue to collect the section
                continue
    
    # Add the last section
    if current_section:
        structural_sections.append(current_section)
    
    # Parse each section
    for section in structural_sections:
        section_text = "\n".join(section)
        section_results = _parse_single_structural_section(section_text)
        
        # Merge results
        for key, value in section_results.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    
    return results

def _parse_single_structural_section(text):
    """Parse a single structural constraint satisfaction table section"""
    results = {}
    lines = text.splitlines()
    
    in_table = False
    
    for line in lines:
        if "| Metric" in line:
            in_table = True
            continue
            
        if in_table and line.strip().startswith("+---"):
            continue
            
        if in_table and line.strip() == "":
            in_table = False
            break
            
        if in_table and "|" in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                metric_name = parts[1].strip()
                value_str = parts[2].strip()
                
                # Extract percentage value
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value_str)
                if m:
                    try:
                        value = float(m.group(1))
                        
                        # Normalize for ring vs cycle wording
                        normalized_metric = metric_name.replace("Ring length", "Cycle length")
                        
                        # Map metric names to our standardized keys
                        if "Ring count 0" in metric_name or "Cycle rank 0" in metric_name:
                            key = "ring_count_0"
                        elif "Ring count 1" in metric_name or "Cycle rank 1" in metric_name:
                            key = "ring_count_1"
                        elif "Ring count 2" in metric_name or "Cycle rank 2" in metric_name:
                            key = "ring_count_2"
                        elif "Ring count 3" in metric_name or "Cycle rank 3" in metric_name:
                            key = "ring_count_3"
                        elif "Ring count 4" in metric_name or "Cycle rank 4" in metric_name:
                            key = "ring_count_4"
                        elif "Ring count 5" in metric_name or "Cycle rank 5" in metric_name:
                            key = "ring_count_5"
                        elif "Ring count 6" in metric_name or "Cycle rank 6" in metric_name:
                            key = "ring_count_6"
                        elif "Ring count 7" in metric_name or "Cycle rank 7" in metric_name:
                            key = "ring_count_7"
                        elif "Ring count 8" in metric_name or "Cycle rank 8" in metric_name:
                            key = "ring_count_8"
                        elif "Ring count 9" in metric_name or "Cycle rank 9" in metric_name:
                            key = "ring_count_9"
                        elif "Ring count 9+" in metric_name or "Cycle rank 9+" in metric_name:
                            key = "ring_count_9plus"
                        elif "Ring count 12" in metric_name or "Cycle rank 12" in metric_name:
                            key = "ring_count_12"
                        elif "Ring count 40" in metric_name or "Cycle rank 40" in metric_name:
                            key = "ring_count_40"
                        elif "Acyclic" in metric_name:
                            key = "acyclic"
                        elif "Cycle length 3" in normalized_metric:
                            key = "ring_length_3"
                        elif "Cycle length 4" in normalized_metric:
                            key = "ring_length_4"
                        elif "Cycle length 5" in normalized_metric:
                            key = "ring_length_5"
                        elif "Cycle length 6" in normalized_metric:
                            key = "ring_length_6"
                        elif "Cycle length 7" in normalized_metric:
                            key = "ring_length_7"
                        elif "Cycle length 8" in normalized_metric:
                            key = "ring_length_8"
                        elif "Cycle length 9" in normalized_metric:
                            key = "ring_length_9"
                        elif "Cycle length 10" in normalized_metric:
                            key = "ring_length_10"
                        elif "Cycle length 11" in normalized_metric:
                            key = "ring_length_11"
                        elif "Cycle length 12" in normalized_metric:
                            key = "ring_length_12"
                        elif "Cycle length >12" in normalized_metric:
                            key = "ring_length_12plus"
                        elif "Ring count >0" in metric_name:
                            key = "ring_count_gt0"
                        elif "Ring count >1" in metric_name:
                            key = "ring_count_gt1"
                        elif "Ring count >2" in metric_name:
                            key = "ring_count_gt2"
                        elif "Ring count >3" in metric_name:
                            key = "ring_count_gt3"
                        elif "Ring count >4" in metric_name:
                            key = "ring_count_gt4"
                        elif "Ring count >5" in metric_name:
                            key = "ring_count_gt5"
                        elif "Ring count >6" in metric_name:
                            key = "ring_count_gt6"
                        elif "Ring count >7" in metric_name:
                            key = "ring_count_gt7"
                        elif "Ring count >9" in metric_name:
                            key = "ring_count_gt9"
                        elif "Ring count >12" in metric_name:
                            key = "ring_count_gt12"
                        elif "Ring count >40" in metric_name:
                            key = "ring_count_gt40"
                        elif "Cycle length >3" in normalized_metric:
                            key = "ring_length_gt3"
                        elif "Cycle length >4" in normalized_metric:
                            key = "ring_length_gt4"
                        elif "Cycle length >5" in normalized_metric:
                            key = "ring_length_gt5"
                        elif "Cycle length >6" in normalized_metric:
                            key = "ring_length_gt6"
                        elif "Cycle length >7" in normalized_metric:
                            key = "ring_length_gt7"
                        elif "Cycle length >8" in normalized_metric:
                            key = "ring_length_gt8"
                        elif "Cycle length >9" in normalized_metric:
                            key = "ring_length_gt9"
                        elif "Cycle length >12" in normalized_metric:
                            key = "ring_length_gt12"
                        else:
                            continue
                            
                        results[key] = value
                    except ValueError:
                        pass
    
    return results

def _parse_fallback_regex(text):
    results = {}
    for key, pat in patterns.items():
        match = re.search(pat, text, flags=re.IGNORECASE)
        if match:
            try:
                if key not in results:
                    results[key] = []
                results[key].append(float(match.group(1)))
            except ValueError:
                pass
    return results

def parse_file(filename):
    with open(filename, "r") as f:
        text = f.read()
    
    # Parse molecular metrics
    results = _parse_table_style(text)
    fallback = _parse_fallback_regex(text)
    for k, v in fallback.items():
        results.setdefault(k, v)
    
    # Parse core metrics
    core_metrics = _parse_core_metrics(text)
    results.update(core_metrics)

    # Parse structural metrics
    structural_results = _parse_structural_table(text)
    results.update(structural_results)
    
    return results

def aggregate(files):
    # Initialize with all possible keys
    all_keys = list(patterns.keys()) + list(STRUCTURAL_PATTERNS.keys()) + list(CORE_METRICS_PATTERNS.keys())
    data = {key: [] for key in all_keys}
    
    for f in files:
        res = parse_file(f)
        for k, v in res.items():
            if k in data:
                if isinstance(v, list):
                    data[k].extend(v)
                else:
                    data[k].append(v)
    
    summary = {}
    for k, vals in data.items():
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            summary[k] = f"{mean} ± {std}"
    
    return summary

def aggregate_with_counts(files):
    """Aggregate data and return both percentages and exact counts"""
    # Initialize with all possible keys
    all_keys = list(patterns.keys()) + list(STRUCTURAL_PATTERNS.keys()) + list(CORE_METRICS_PATTERNS.keys())
    data = {key: [] for key in all_keys}
    
    for f in files:
        res = parse_file(f)
        for k, v in res.items():
            if k in data:
                if isinstance(v, list):
                    data[k].extend(v)
                else:
                    data[k].append(v)
    
    summary_percent = {}
    summary_counts = {}
    
    for k, vals in data.items():
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            
            # Handle molecules_generated specially - it's not a percentage
            if k == "molecules_generated":
                # Show total count across all seeds (same as other metrics)
                mean_total = mean * 1000 * 5  # Convert from thousands to actual count, then multiply by 5 seeds
                std_total = std * 1000 * 5
                summary_counts[k] = f"{mean_total} ± {std_total}"
                continue
            
            summary_percent[k] = f"{mean} ± {std}"
            
            # Convert percentages to counts (assuming 10,000 molecules per seed, 5 seeds total)
            mean_count = mean * 500  # 10,000 * 5 seeds / 100 for percentage
            std_count = std * 500
            
            # Format counts as integers if they're close to zero
            if mean_count < 5.0:
                summary_counts[k] = "0 ± 0"
            else:
                summary_counts[k] = f"{mean_count} ± {std_count}"
    
    return summary_percent, summary_counts

def generate_latex_tables(summary_data):
    """Generate LaTeX tables for structural constraint satisfaction"""
    latex_content = []
    
    # Cycle rank distribution table
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Constraint} & $0$ & $1$ & $2$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$ & $9+$ \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    constraints = [
        ("no_constraint", "No Constraint"),
        ("ring_count_0", "Ring Count ≤ 0"),
        ("ring_count_1", "Ring Count ≤ 1"),
        ("ring_count_2", "Ring Count ≤ 2"),
        ("ring_count_3", "Ring Count ≤ 3"),
        ("ring_count_4", "Ring Count ≤ 4"),
        ("ring_count_5", "Ring Count ≤ 5"),
        ("ring_count_6", "Ring Count ≤ 6"),
        ("ring_count_7", "Ring Count ≤ 7"),
        ("ring_count_8", "Ring Count ≤ 8"),
        ("ring_count_9", "Ring Count ≤ 9"),
        ("ring_count_12", "Ring Count ≤ 12"),
        ("ring_count_40", "Ring Count ≤ 40"),
    ]
    
    for constraint_key, constraint_name in constraints:
        if constraint_key not in summary_data:
            continue
        
        row = f"        {constraint_name}"
        for rank in range(10):
            if rank == 9:
                key = f"ring_count_9plus"
            else:
                key = f"ring_count_{rank}"
            
            if key in summary_data[constraint_key]:
                row += f" & {summary_data[constraint_key][key]}"
            else:
                row += " & 0 ± 0"
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Ring count distribution across different constraint configurations.}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # Cycle length distribution table
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Ring Length} & $0$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$ \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    length_constraints = [
        ("no_constraint", "No Constraint"),
        ("ring_length_3", "Ring Length ≤ 3"),
        ("ring_length_4", "Ring Length ≤ 4"),
        ("ring_length_5", "Ring Length ≤ 5"),
        ("ring_length_6", "Ring Length ≤ 6"),
        ("ring_length_7", "Ring Length ≤ 7"),
        ("ring_length_8", "Ring Length ≤ 8"),
        ("ring_length_9", "Ring Length ≤ 9"),
        ("ring_length_12", "Ring Length ≤ 12"),
    ]
    
    for constraint_key, constraint_name in length_constraints:
        if constraint_key not in summary_data:
            continue
        
        row = f"        {constraint_name}"
        
        # Acyclic (0)
        if "acyclic" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['acyclic']}"
        else:
            row += " & 0.0 ± 0.0"
        
        # Cycle lengths 3-8
        for length in [3, 4, 5, 6, 7, 8]:
            key = f"ring_length_{length}"
            if key in summary_data[constraint_key]:
                row += f" & {summary_data[constraint_key][key]}"
            else:
                row += " & 0 ± 0"
        
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Ring length distribution (including acyclic) across different constraint configurations.}")
    latex_content.append("\\end{table}")
    
    return "\n".join(latex_content)

def generate_latex_tables_with_counts(summary_data):
    """Generate LaTeX tables for structural constraint satisfaction with both percentages and counts"""
    latex_content = []
    
    # Cycle rank distribution table - Percentages
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Constraint} & $0$ & $1$ & $2$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$ & $9+$ \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    constraints = [
        ("no_constraint", "No Constraint"),
        ("ring_count_0", "Ring Count ≤ 0"),
        ("ring_count_1", "Ring Count ≤ 1"),
        ("ring_count_2", "Ring Count ≤ 2"),
        ("ring_count_3", "Ring Count ≤ 3"),
        ("ring_count_4", "Ring Count ≤ 4"),
        ("ring_count_5", "Ring Count ≤ 5"),
        ("ring_count_6", "Ring Count ≤ 6"),
        ("ring_count_7", "Ring Count ≤ 7"),
        ("ring_count_8", "Ring Count ≤ 8"),
        ("ring_count_9", "Ring Count ≤ 9"),
        ("ring_count_12", "Ring Count ≤ 12"),
        ("ring_count_40", "Ring Count ≤ 40"),
    ]
    
    for constraint_key, constraint_name in constraints:
        if constraint_key not in summary_data:
            continue
        
        row = f"        {constraint_name}"
        for rank in range(10):
            if rank == 9:
                key = f"ring_count_9plus"
            else:
                key = f"ring_count_{rank}"
            
            if key in summary_data[constraint_key]:
                row += f" & {summary_data[constraint_key][key]}"
            else:
                row += " & 0.0000 ± 0.0000"
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Cycle rank distribution (\\%) across different constraint configurations.}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # Cycle length distribution table - Percentages
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Ring Length} & $0$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$ \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    length_constraints = [
        ("no_constraint", "No Constraint"),
        ("ring_length_3", "Ring Length ≤ 3"),
        ("ring_length_4", "Ring Length ≤ 4"),
        ("ring_length_5", "Ring Length ≤ 5"),
        ("ring_length_6", "Ring Length ≤ 6"),
        ("ring_length_7", "Ring Length ≤ 7"),
        ("ring_length_8", "Ring Length ≤ 8"),
        ("ring_length_9", "Ring Length ≤ 9"),
        ("ring_length_12", "Ring Length ≤ 12"),
    ]
    
    for constraint_key, constraint_name in length_constraints:
        if constraint_key not in summary_data:
            continue
        
        row = f"        {constraint_name}"
        
        # Acyclic (0)
        if "acyclic" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['acyclic']}"
        else:
            row += " & 0 ± 0"
        
        # Cycle lengths 3-8
        for length in [3, 4, 5, 6, 7, 8]:
            key = f"ring_length_{length}"
            if key in summary_data[constraint_key]:
                row += f" & {summary_data[constraint_key][key]}"
            else:
                row += " & 0 ± 0"
        
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Ring length distribution (\\%) across different constraint configurations.}")
    latex_content.append("\\end{table}")
    
    return "\n".join(latex_content)

def generate_latex_tables_counts(summary_data):
    """Generate LaTeX tables for structural constraint satisfaction with molecule counts"""
    latex_content = []
    
    # Cycle rank distribution table - Counts
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Constraint} & $0$ & $1$ & $2$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$ & $9+$ \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    constraints = [
        ("no_constraint", "No Constraint"),
        ("ring_count_0", "Ring Count ≤ 0"),
        ("ring_count_1", "Ring Count ≤ 1"),
        ("ring_count_2", "Ring Count ≤ 2"),
        ("ring_count_3", "Ring Count ≤ 3"),
        ("ring_count_4", "Ring Count ≤ 4"),
        ("ring_count_5", "Ring Count ≤ 5"),
        ("ring_count_6", "Ring Count ≤ 6"),
        ("ring_count_7", "Ring Count ≤ 7"),
        ("ring_count_8", "Ring Count ≤ 8"),
        ("ring_count_9", "Ring Count ≤ 9"),
        ("ring_count_12", "Ring Count ≤ 12"),
        ("ring_count_40", "Ring Count ≤ 40"),
    ]
    
    for constraint_key, constraint_name in constraints:
        if constraint_key not in summary_data:
            continue
        
        row = f"        {constraint_name}"
        for rank in range(10):
            if rank == 9:
                key = f"ring_count_9plus"
            else:
                key = f"ring_count_{rank}"
            
            if key in summary_data[constraint_key]:
                row += f" & {summary_data[constraint_key][key]}"
            else:
                row += " & 0 ± 0"
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Ring count distribution (molecule counts) across different constraint configurations.}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # Cycle length distribution table - Counts
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Ring Length} & $0$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$ \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    length_constraints = [
        ("no_constraint", "No Constraint"),
        ("ring_length_3", "Ring Length ≤ 3"),
        ("ring_length_4", "Ring Length ≤ 4"),
        ("ring_length_5", "Ring Length ≤ 5"),
        ("ring_length_6", "Ring Length ≤ 6"),
        ("ring_length_7", "Ring Length ≤ 7"),
        ("ring_length_8", "Ring Length ≤ 8"),
        ("ring_length_9", "Ring Length ≤ 9"),
        ("ring_length_12", "Ring Length ≤ 12"),
    ]
    
    for constraint_key, constraint_name in length_constraints:
        if constraint_key not in summary_data:
            continue
        
        row = f"        {constraint_name}"
        
        # Acyclic (0)
        if "acyclic" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['acyclic']}"
        else:
            row += " & 0 ± 0"
        
        # Cycle lengths 3-8
        for length in [3, 4, 5, 6, 7, 8]:
            key = f"ring_length_{length}"
            if key in summary_data[constraint_key]:
                row += f" & {summary_data[constraint_key][key]}"
            else:
                row += " & 0 ± 0"
        
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Ring length distribution (including acyclic, molecule counts) across different constraint configurations.}")
    latex_content.append("\\end{table}")
    
    return "\n".join(latex_content)

def generate_core_metrics_tables(summary_data):
    """Generate LaTeX tables for core sampling metrics"""
    latex_content = []
    
    # Core metrics table - Percentages
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Constraint} & \\textbf{FCD} & \\textbf{Unique} & \\textbf{Novel} & \\textbf{Valid} & \\textbf{Disconnected} & \\textbf{Property satisfied} & \\textbf{V.U.N.} \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    constraints = [
        ("no_constraint", "No Constraint"),
        ("planar", "Planar"),
        ("ring_count_0", "Ring Count ≤ 0"),
        ("ring_count_1", "Ring Count ≤ 1"),
        ("ring_count_2", "Ring Count ≤ 2"),
        ("ring_count_3", "Ring Count ≤ 3"),
        ("ring_count_4", "Ring Count ≤ 4"),
        ("ring_count_5", "Ring Count ≤ 5"),
        ("ring_count_6", "Ring Count ≤ 6"),
        ("ring_count_7", "Ring Count ≤ 7"),
        ("ring_count_8", "Ring Count ≤ 8"),
        ("ring_count_9", "Ring Count ≤ 9"),
        ("ring_count_12", "Ring Count ≤ 12"),
        ("ring_count_40", "Ring Count ≤ 40"),
        ("ring_length_3", "Ring Length ≤ 3"),
        ("ring_length_4", "Ring Length ≤ 4"),
        ("ring_length_5", "Ring Length ≤ 5"),
        ("ring_length_6", "Ring Length ≤ 6"),
        ("ring_length_7", "Ring Length ≤ 7"),
        ("ring_length_8", "Ring Length ≤ 8"),
        ("ring_length_9", "Ring Length ≤ 9"),
        ("ring_length_12", "Ring Length ≤ 12"),
    ]
    
    for constraint_key, constraint_name in constraints:
        if constraint_key not in summary_data:
            continue
            
        row = f"        {constraint_name}"
        
        # FCD
        if "FCD" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['FCD']}"
        else:
            row += " & 0 ± 0"
        
        # Unique
        if "Unique" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Unique']}"
        else:
            row += " & 0 ± 0"
        
        # Novel
        if "Novel" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Novel']}"
        else:
            row += " & 0 ± 0"
        
        # Valid
        if "Valid" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Valid']}"
        else:
            row += " & 0 ± 0"
        
        # Disconnected
        if "Disconnected" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Disconnected']}"
        else:
            row += " & 0 ± 0"
        
        # Property satisfied
        if "Property_satisfied" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Property_satisfied']}"
        else:
            row += " & 0 ± 0"
        
        # VUN
        if "VUN" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['VUN']}"
        else:
            row += " & 0 ± 0"
        
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Core sampling metrics (\\%) across different constraint configurations.}")
    latex_content.append("\\end{table}")
    
    return "\n".join(latex_content)

def generate_core_metrics_tables_counts(summary_data):
    """Generate LaTeX tables for core sampling metrics with molecule counts"""
    latex_content = []
    
    # Core metrics table - Counts
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\centering")
    latex_content.append("    \\begin{tabular}{l c c c c c c c}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Constraint} & \\textbf{FCD} & \\textbf{Unique} & \\textbf{Novel} & \\textbf{Valid} & \\textbf{Disconnected} & \\textbf{Property satisfied} & \\textbf{V.U.N.} \\\\")
    latex_content.append("        \\midrule")
    
    # Add rows for each constraint type
    constraints = [
        ("no_constraint", "No Constraint"),
        ("planar", "Planar"),
        ("ring_count_0", "Ring Count ≤ 0"),
        ("ring_count_1", "Ring Count ≤ 1"),
        ("ring_count_2", "Ring Count ≤ 2"),
        ("ring_count_3", "Ring Count ≤ 3"),
        ("ring_count_4", "Ring Count ≤ 4"),
        ("ring_count_5", "Ring Count ≤ 5"),
        ("ring_count_6", "Ring Count ≤ 6"),
        ("ring_count_7", "Ring Count ≤ 7"),
        ("ring_count_8", "Ring Count ≤ 8"),
        ("ring_count_9", "Ring Count ≤ 9"),
        ("ring_count_12", "Ring Count ≤ 12"),
        ("ring_count_40", "Ring Count ≤ 40"),
        ("ring_length_3", "Ring Length ≤ 3"),
        ("ring_length_4", "Ring Length ≤ 4"),
        ("ring_length_5", "Ring Length ≤ 5"),
        ("ring_length_6", "Ring Length ≤ 6"),
        ("ring_length_7", "Ring Length ≤ 7"),
        ("ring_length_8", "Ring Length ≤ 8"),
        ("ring_length_9", "Ring Length ≤ 9"),
        ("ring_length_12", "Ring Length ≤ 12"),
    ]
    
    for constraint_key, constraint_name in constraints:
        if constraint_key not in summary_data:
            continue
            
        row = f"        {constraint_name}"
        
        # FCD (keep as is, not a count)
        if "FCD" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['FCD']}"
        else:
            row += " & 0 ± 0"
        
        # Unique count
        if "Unique" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Unique']}"
        else:
            row += " & 0 ± 0"
        
        # Novel count
        if "Novel" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Novel']}"
        else:
            row += " & 0 ± 0"
        
        # Valid count
        if "Valid" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Valid']}"
        else:
            row += " & 0 ± 0"
        
        # Disconnected count
        if "Disconnected" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Disconnected']}"
        else:
            row += " & 0 ± 0"
        
        # Property satisfied count
        if "Property_satisfied" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['Property_satisfied']}"
        else:
            row += " & 0 ± 0"
        
        # VUN count
        if "VUN" in summary_data[constraint_key]:
            row += f" & {summary_data[constraint_key]['VUN']}"
        else:
            row += " & 0 ± 0"
        
        row += " \\\\"
        latex_content.append(row)
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("    \\caption{Core sampling metrics (molecule counts) across different constraint configurations.}")
    latex_content.append("\\end{table}")
    
    return "\n".join(latex_content)

def generate_report() -> str:
    lines = []
    
    # Process all constraint types
    constraint_types = [
        "no_constraint",
        "planar",
        "ring_count_0",
        "ring_count_1", 
        "ring_count_2",
        "ring_count_3",
        "ring_count_4",
        "ring_count_5",
        "ring_count_6",
        "ring_count_7",
        "ring_count_8",
        "ring_count_9",
        "ring_count_12",
        "ring_count_40",
        "ring_length_3",
        "ring_length_4",
        "ring_length_5",
        "ring_length_6",
        "ring_length_7",
        "ring_length_8",
        "ring_length_9",
        "ring_length_12",
    ]
    
    all_summaries = {}
    all_summaries_counts = {}
    
    for constraint in constraint_types:
        files = glob.glob(f"../simple_cycle_done/test_{constraint}_*.out")
        if not files:
            continue
        
        lines.append("")
        lines.append(f"=== {constraint} ===")
        summary_percent, summary_counts = aggregate_with_counts(files)
        # Remove redundant 'Satisfied' metric from summaries
        if 'Satisfied' in summary_percent:
            del summary_percent['Satisfied']
        if 'Satisfied' in summary_counts:
            del summary_counts['Satisfied']
        all_summaries[constraint] = summary_percent
        all_summaries_counts[constraint] = summary_counts
        lines.append("Percentages: " + str(summary_percent))
        lines.append("Counts: " + str(summary_counts))
        
        # Ring length distribution section removed for cleaner aggregate output
    
    return "\n".join(lines).rstrip() + "\n", all_summaries, all_summaries_counts

if __name__ == "__main__":
    report, summaries, summaries_counts = generate_report()
    print(report)
    
    # Save text report
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "aggregate_structural_summary.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Saved aggregate summary to: {out_path}")
    
    # Generate and save LaTeX tables with 4 decimal places
    latex_content = generate_latex_tables_with_counts(summaries)
    latex_path = os.path.join(out_dir, "structural_constraint_tables.tex")
    with open(latex_path, "w") as f:
        f.write(latex_content)
    print(f"Saved LaTeX tables to: {latex_path}")
    
    # Generate and save count tables
    count_latex_content = generate_latex_tables_counts(summaries_counts)
    count_latex_path = os.path.join(out_dir, "structural_constraint_tables_counts.tex")
    with open(count_latex_path, "w") as f:
        f.write(count_latex_content)
    print(f"Saved count LaTeX tables to: {count_latex_path}")
    
    # Generate and save core metrics tables
    core_metrics_content = generate_core_metrics_tables(summaries)
    core_metrics_path = os.path.join(out_dir, "core_metrics_tables.tex")
    with open(core_metrics_path, "w") as f:
        f.write(core_metrics_content)
    print(f"Saved core metrics LaTeX tables to: {core_metrics_path}")
    
    # Generate and save core metrics count tables
    core_metrics_counts_content = generate_core_metrics_tables_counts(summaries_counts)
    core_metrics_counts_path = os.path.join(out_dir, "core_metrics_tables_counts.tex")
    with open(core_metrics_counts_path, "w") as f:
        f.write(core_metrics_counts_content)
    print(f"Saved core metrics count LaTeX tables to: {core_metrics_counts_path}") 