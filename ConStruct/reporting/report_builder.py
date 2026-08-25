import os, statistics
import logging
from typing import Dict, Any, List, Optional, Tuple

# -------- formatting (no arrows, consistent precision) --------
def pct(x: Optional[float]) -> str:
    if x is None: return "NA"
    if 0.0 <= x <= 1.0: x *= 100.0
    # Use 4 decimal places for structural enforcement values
    return f"{round(x, 4)}%"

def f3(x: Optional[float]) -> str:
    return "NA" if x is None else f"{round(x, 3)}"

# --- new: helper function to setup additional logging to main.log ---
def setup_additional_logging() -> Optional[logging.Logger]:
    """
    Set up additional logging to the main.log file in the outputs directory.
    Returns the logger if successful, None otherwise.
    """
    try:
        # Get current working directory (should be project root due to chdir: False)
        cwd = os.getcwd()
        
        # Look for outputs directory and find the most recent run directory
        outputs_dir = os.path.join(cwd, "outputs")
        if not os.path.exists(outputs_dir):
            return None
            
        # Find the most recent date directory
        date_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
        if not date_dirs:
            return None
            
        latest_date = max(date_dirs)
        date_path = os.path.join(outputs_dir, latest_date)
        
        # Find the most recent run directory within the date directory
        run_dirs = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
        if not run_dirs:
            return None
            
        latest_run = max(run_dirs)
        run_path = os.path.join(date_path, latest_run)
        
        # Check if main.log exists
        main_log_path = os.path.join(run_path, "main.log")
        if not os.path.exists(main_log_path):
            return None
            
        # Create a logger for additional logging
        logger = logging.getLogger("report_builder_additional")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Create file handler for main.log
        file_handler = logging.FileHandler(main_log_path, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter (no timestamp since we're appending to existing log)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False  # Prevent propagation to root logger
        
        return logger
        
    except Exception as e:
        # Silently fail - don't interrupt the main functionality
        return None

def log_and_print(message: str, logger: Optional[logging.Logger] = None):
    """
    Print message to console and optionally log to main.log file.
    """
    print(message)
    if logger:
        logger.info(message)

# --- new: robust scalar coercion for hist/arrays/tensors/W&B Histogram ---
def to_scalar(value) -> Optional[float]:
    """
    Convert metric payloads (float/int/list/tuple/np.ndarray/torch.Tensor/dict/W&B Histogram-like)
    into a single scalar suitable for table printing. Returns None if not possible.
    For histograms (e.g., abs_diff_deg_hist), we return the L1 sum (sum of absolute bins).
    """
    if value is None:
        return None
    # numeric fast-path
    if isinstance(value, (int, float)):
        return float(value)
    # torch
    try:
        import torch  # type: ignore
        if isinstance(value, torch.Tensor):
            return float(value.detach().abs().sum().cpu().item())
    except Exception:
        pass
    # numpy
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.ndarray):
            return float(np.abs(value).sum())
    except Exception:
        pass
    # list/tuple -> sum abs
    if isinstance(value, (list, tuple)):
        try:
            return float(sum(abs(float(v)) for v in value))
        except Exception:
            return None
    # dict -> sum abs of values
    if isinstance(value, dict):
        if "counts" in value:
            counts = value["counts"]
            return float(sum(abs(float(v)) for v in counts))
        try:
            return float(sum(abs(float(v)) for v in value.values()))
        except Exception:
            return None
    # W&B Histogram-like: has .histogram (list of counts) or .to_json()
    try:
        if hasattr(value, "histogram"):
            h = getattr(value, "histogram", None)
            if isinstance(h, (list, tuple)):
                return float(sum(abs(float(v)) for v in h))
        if hasattr(value, "to_json"):
            j = value.to_json()
            h = j.get("histogram") if isinstance(j, dict) else None
            if isinstance(h, (list, tuple)):
                return float(sum(abs(float(v)) for v in h))
    except Exception:
        pass
    # last resort: try float cast
    try:
        return float(value)
    except Exception:
        return None

def f2(x: Optional[float]) -> str:
    return "NA" if x is None else f"{round(x, 2)}"

def f1(x: Optional[float]) -> str:
    return "NA" if x is None else f"{round(x, 1)}"

def intc(x: Optional[int]) -> str:
    return "NA" if x is None else f"{int(x):,}"

def arr_pct_str(counts: Optional[List[int]], total: Optional[int], take: int) -> str:
    if not counts or total in (None,0) or len(counts) < take: return "NA"
    return " / ".join([f"{round(100.0*c/total, 4)}" for c in counts[:take]])

def split_prefix(split: str) -> str:
    assert split in {"val","test"}
    return f"{split}_sampling"

# -------- constraint detection & captions --------
def detect_constraint_kind(metrics: Dict[str, Any], split: str, cfg=None) -> str:
    if cfg is not None and hasattr(cfg.model, "rev_proj"):
        projector = cfg.model.rev_proj
        if projector in {
            "ring_count_at_most",
            "ring_count_at_least",
            "ring_length_at_most",
            "ring_length_at_least",
            "planar",
        }:
            return "planarity" if projector == "planar" else projector
    return "none"


def constraint_caption(kind: str, meta: Dict[str, Any]) -> str:
    if kind == "ring_count_at_most":
        return f"Ring count ≤ {meta.get('max_rings', 'k')}"
    if kind == "ring_count_at_least":
        return f"Ring count ≥ {meta.get('min_rings', 'k')}"
    if kind == "ring_length_at_most":
        return f"Maximum ring length ≤ {meta.get('max_ring_length', 'L')}"
    if kind == "ring_length_at_least":
        return f"Maximum ring length ≥ {meta.get('min_ring_length', 'L')}"
    if kind == "planarity":
        return "Planar molecules only"
    return "No structural constraint"

# -------- Core (main) --------
def collect_core(split: str, metrics: Dict[str, Any], N_total: Optional[int], cfg=None) -> List[Tuple[str,str]]:
    p = split_prefix(split)
    rows = []
    rows.append(("Molecules generated (N)", intc(N_total)))
    rows.append(("FCD",                 f3(metrics.get(f"{p}/fcd score"))))
    rows.append(("Unique (%)",          pct(metrics.get(f"{p}/Uniqueness"))))
    rows.append(("Novel (%)",           pct(metrics.get(f"{p}/Novelty"))))
    rows.append(("Valid (%)",           pct(metrics.get(f"{p}/Validity"))))
    rows.append(("Disconnected (%)",    pct(metrics.get(f"{p}/Disconnected"))))
    # Property satisfied (%)
    kind = detect_constraint_kind(metrics, split, cfg)
    prop = None
    if kind == "planarity":   prop = metrics.get(f"{p}/planarity")
    elif kind in {"ring_count_at_most", "ring_count_at_least"}:
        prop = metrics.get(f"{p}/ring_count_satisfaction")
    elif kind in {"ring_length_at_most", "ring_length_at_least"}:
        prop = metrics.get(f"{p}/ring_length_satisfaction")
    rows.append(("Property satisfied (%)", "—" if kind=="none" else pct(prop)))
    # V.U.N. (%), over generated set
    u = metrics.get(f"{p}/Uniqueness"); n = metrics.get(f"{p}/Novelty"); v = metrics.get(f"{p}/Validity")
    if None not in (u,n,v):
        if u>1: u/=100.0
        if n>1: n/=100.0
        if v>1: v/=100.0
        rows.append(("V.U.N. (%)", f"{round(100.0*(u*n*v), 4)}%"))
    return rows

def core_definitions_md() -> str:
    return (
        "\n**Definitions**: "
        "**FCD** — Fréchet ChemNet Distance computed on valid canonical SMILES; lower is better. "
        "**Unique/Novel/Valid** — proportions computed over all generated molecules (not only valid). "
        "**Disconnected** — share of graphs with >1 connected component. "
        "**Property satisfied** — share of generated graphs meeting the configured structural target. "
        "**V.U.N.** — product of Valid × Unique × Novel (in [0,100]%).\n"
    )

# -------- Structural (main) --------
def collect_structural(
    split: str,
    metrics: Dict[str, Any],
    N_total: Optional[int],
    ring_count_counts: Optional[List[int]] = None,
    ring_length_counts: Optional[List[int]] = None,
    max_rings: Optional[int] = None,
    max_ring_length: Optional[int] = None,
    min_rings: Optional[int] = None,
    min_ring_length: Optional[int] = None,
    cfg=None,
) -> List[Tuple[str, str]]:
    p = split_prefix(split)
    rows = []
    kind = detect_constraint_kind(metrics, split, cfg)

    if kind == "planarity":
        rows.append(("Planarity satisfied (%)", pct(metrics.get(f"{p}/planarity"))))
    elif kind.startswith("ring_count_"):
        rows.append(("Ring count satisfied (%)", pct(metrics.get(f"{p}/ring_count_satisfaction"))))
    elif kind.startswith("ring_length_"):
        rows.append(("Ring length satisfied (%)", pct(metrics.get(f"{p}/ring_length_satisfaction"))))
    else:
        rows.append(("Constraint", "No structural constraint"))

    def add_ring_count(index, count):
        label = f"Ring count {index} (%)" if index < 9 else "Ring count 9+ (%)"
        rows.append((label, f"{round(100.0 * count / N_total, 4) if N_total else 0.0}%"))

    def ring_length_count(length):
        if length == 0:
            return ring_length_counts[0] if ring_length_counts else 0
        index = (length - 3) + 1
        return ring_length_counts[index] if ring_length_counts and index < len(ring_length_counts) else 0

    if kind == "ring_count_at_most" and ring_count_counts is not None:
        threshold = int(max_rings)
        for index in range(threshold + 1):
            add_ring_count(index, ring_count_counts[index] if index < len(ring_count_counts) else 0)
        violating = sum(ring_count_counts[threshold + 1:])
        rows.append((f"Ring count >{threshold} (%)", f"{round(100.0 * violating / N_total, 4) if N_total else 0.0}%"))
    elif kind == "ring_count_at_least" and ring_count_counts is not None:
        threshold = int(min_rings)
        violating = sum(ring_count_counts[:threshold])
        rows.append((f"Ring count <{threshold} (%)", f"{round(100.0 * violating / N_total, 4) if N_total else 0.0}%"))
        for index in range(threshold, len(ring_count_counts)):
            add_ring_count(index, ring_count_counts[index])
    elif kind == "ring_length_at_most" and ring_length_counts is not None:
        threshold = int(max_ring_length)
        rows.append(("Acyclic (max len 0) (%)", f"{round(100.0 * ring_length_count(0) / N_total, 4) if N_total else 0.0}%"))
        for length in range(3, threshold + 1):
            count = ring_length_count(length)
            rows.append((f"Ring length {length} (max) (%)", f"{round(100.0 * count / N_total, 4) if N_total else 0.0}%"))
        first_violating_index = (threshold - 3) + 2
        violating = sum(ring_length_counts[first_violating_index:])
        rows.append((f"Ring length >{threshold} (max) (%)", f"{round(100.0 * violating / N_total, 4) if N_total else 0.0}%"))
    elif kind == "ring_length_at_least" and ring_length_counts is not None:
        threshold = int(min_ring_length)
        violating = ring_length_count(0) + sum(ring_length_count(length) for length in range(3, threshold))
        rows.append((f"Maximum ring length <{threshold} (%)", f"{round(100.0 * violating / N_total, 4) if N_total else 0.0}%"))
        for length in range(threshold, 13):
            count = ring_length_count(length)
            rows.append((f"Ring length {length} (max) (%)", f"{round(100.0 * count / N_total, 4) if N_total else 0.0}%"))
        rows.append(("Ring length >12 (max) (%)", f"{round(100.0 * ring_length_counts[-1] / N_total, 4) if N_total else 0.0}%"))
    elif kind == "none":
        if ring_count_counts is not None:
            for index, count in enumerate(ring_count_counts):
                add_ring_count(index, count)
        if ring_length_counts is not None:
            rows.append(("Acyclic (max len 0) (%)", f"{round(100.0 * ring_length_count(0) / N_total, 4) if N_total else 0.0}%"))
            for length in range(3, 13):
                count = ring_length_count(length)
                rows.append((f"Ring length {length} (max) (%)", f"{round(100.0 * count / N_total, 4) if N_total else 0.0}%"))
            rows.append(("Ring length >12 (max) (%)", f"{round(100.0 * ring_length_counts[-1] / N_total, 4) if N_total else 0.0}%"))
    return rows


# -------- Alignment (appendix) --------
def collect_alignment(split: str, metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
    p = split_prefix(split)
    rows = []
    rows.append(("NumNodes W1", f3(to_scalar(metrics.get(f"{p}/NumNodesW1")))))
    rows.append(("NodeTypes TV", f3(to_scalar(metrics.get(f"{p}/NodeTypesTV")))))
    rows.append(("EdgeTypes TV", f3(to_scalar(metrics.get(f"{p}/EdgeTypesTV")))))
    if f"{p}/abs_diff_deg_hist" in metrics:
        deg_l1 = to_scalar(metrics.get(f"{p}/abs_diff_deg_hist"))
        rows.append(("Degree hist |Δ| (L1 sum)", f3(deg_l1)))
    return rows


# -------- Chemistry (appendix; from valid SMILES, optional) --------
def chemistry_from_smiles(smiles: List[str]) -> Dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return {}
    mols = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(s)
            if m is not None: mols.append(m)
        except Exception:
            pass
    if not mols: return {}
    mw = [Descriptors.MolWt(m) for m in mols]
    heavy = [m.GetNumHeavyAtoms() for m in mols]
    comps = {"C":0,"N":0,"O":0,"F":0}
    total_atoms = 0
    for m in mols:
        for a in m.GetAtoms():
            sym = a.GetSymbol()
            if sym in comps: comps[sym]+=1
            total_atoms += 1
    comp_pct = {k: (100.0*v/total_atoms) if total_atoms>0 else 0.0 for k,v in comps.items()}
    def meanstd(x):
        return (statistics.mean(x), statistics.pstdev(x) if len(x)>1 else 0.0)
    mw_mu, mw_sd = meanstd(mw)
    hv_mu, hv_sd = meanstd(heavy)
    return {
        "N_valid_for_chem": len(mols),
        "MW_mean": mw_mu, "MW_sd": mw_sd,
        "Heavy_mean": hv_mu, "Heavy_sd": hv_sd,
        "Comp_C": comp_pct["C"], "Comp_N": comp_pct["N"], "Comp_O": comp_pct["O"], "Comp_F": comp_pct["F"],
    }

def collect_chemistry(chem: Dict[str, Any]) -> List[Tuple[str,str]]:
    if not chem: return []
    return [
        ("Valid molecules used (chem)", intc(chem.get("N_valid_for_chem"))),
        ("Molecular weight (Da) mean ± sd", f"{f1(chem.get('MW_mean'))} ± {f1(chem.get('MW_sd'))}"),
        ("Heavy atoms mean ± sd",            f"{f1(chem.get('Heavy_mean'))} ± {f1(chem.get('Heavy_sd'))}"),
        ("Atom composition % (C/N/O/F)",     f"{f1(chem.get('Comp_C'))}/{f1(chem.get('Comp_N'))}/{f1(chem.get('Comp_O'))}/{f1(chem.get('Comp_F'))}"),
    ]

# -------- Timing (appendix) --------
# Align with ConStruct: report sampling time for N graphs and per-graph time (ms).
def collect_timing(t: Dict[str, Any], N_total: Optional[int], device_meta: Dict[str, str]) -> List[Tuple[str,str]]:
    rows = []
    samp_s = t.get("sampling_sec")         # total sampling seconds for this split
    rows.append(("Sampling time (s) for N", f"{f2(samp_s)} for N={intc(N_total)}"))
    per_ms = None
    if samp_s is not None and N_total not in (None, 0):
        per_ms = 1000.0 * float(samp_s) / int(N_total)
    rows.append(("Per-graph sampling time (ms)", f1(per_ms)))
    rows.append(("Sampling batches",            f2(t.get("sampling_batches")) if t.get("sampling_batches") is not None else "NA"))
    
    # Handle projection metrics with better defaults
    proj_ms_mean = t.get("proj_ms_mean")
    if proj_ms_mean is not None and proj_ms_mean == 0.0:
        rows.append(("Projector mean time (ms)", "0.0 (no projection)"))
    else:
        rows.append(("Projector mean time (ms)", f1(proj_ms_mean)))
    
    proj_share_pct = t.get("proj_share_pct")
    if proj_share_pct is not None and proj_share_pct == 0.0:
        rows.append(("Projector share of sampling (%)", "0.0% (no projection)"))
    else:
        rows.append(("Projector share of sampling (%)", f1(proj_share_pct)))
    
    rows.append(("Wall-clock elapsed (hh:mm)",  t.get("wall_clock_hhmm","NA")))
    # Device/context for reproducibility
    dev = device_meta or {}
    ctx = ", ".join([f"{k}={v}" for k,v in dev.items() if v])
    rows.append(("Environment", ctx if ctx else "NA"))
    return rows

# -------- Renderers (titles self-contained) --------
def render_table_md(title: str, subtitle: str, rows: List[Tuple[str,str]], tail_note: Optional[str]=None) -> str:
    s = [f"### {title}", f"{subtitle}", "", "| Metric | Value |", "|---|---:|"]
    for k,v in rows: s.append(f"| {k} | {v} |")
    if tail_note: s.append(tail_note)
    s.append("")  # newline
    return "\n".join(s)

def render_table_console(title: str, subtitle: str, rows: List[Tuple[str,str]], tail_note: Optional[str]=None) -> str:
    """Render a nicely formatted table for console output with proper alignment."""
    if not rows:
        return f"{title}\n{subtitle}\n(No data available)"
    
    # Find the maximum width for the metric column
    max_metric_width = max(len(str(k)) for k, _ in rows)
    
    # Set a reasonable maximum width for values to prevent very wide tables
    max_value_width = max(len(str(v)) for _, v in rows)
    value_width = min(50, max(20, max_value_width + 2))  # Between 20 and 50 characters
    
    # Create the table
    lines = []
    lines.append(title)
    lines.append(subtitle)
    lines.append("")
    
    # Create separator line
    separator = "+" + "-" * (max_metric_width + 2) + "+" + "-" * value_width + "+"
    lines.append(separator)
    
    # Header
    header = f"| {('Metric').ljust(max_metric_width)} | {'Value'.ljust(value_width)} |"
    lines.append(header)
    lines.append(separator)
    
    # Data rows
    for metric, value in rows:
        metric_str = str(metric).ljust(max_metric_width)
        value_str = str(value)
        
        # Truncate long values and add ellipsis
        if len(value_str) > value_width - 1:
            value_str = value_str[:value_width-4] + "..."
        
        value_str = value_str.ljust(value_width)
        row = f"| {metric_str} | {value_str} |"
        lines.append(row)
    
    lines.append(separator)
    
    # Add tail note if provided
    if tail_note:
        lines.append("")
        lines.append(tail_note)
    
    lines.append("")  # final newline
    return "\n".join(lines)

def write_tables(outdir: str, split: str, exp_name: str, dataset: str, constraint_str: str,
                 N_total: Optional[int], diffusion_steps: Optional[int],
                 eval_meta: Dict[str, Any],
                 core: List[Tuple[str,str]],
                 structural: List[Tuple[str,str]],
                 alignment: List[Tuple[str,str]],
                 timing: List[Tuple[str,str]],
                 include_alignment: bool = True,
                 include_timing: bool = True):
    os.makedirs(outdir, exist_ok=True)
    hdr = f"{exp_name} — {split.upper()} — {dataset}"
    sub = f"Constraint: {constraint_str} · N={intc(N_total)} · d={intc(diffusion_steps)} · sampler={eval_meta.get('sampler','reverse')}"
    # Structural subtitle explains the calculation:
    sub_struct = (
        sub
        + " · Structural cycles via unique simple cycles; "
          "ring-length rows = per-molecule MAX ring length; "
          "aromatic bonds included"
    )
    with open(os.path.join(outdir, f"{split}_core.md"), "w") as f:
        f.write(render_table_md(hdr, sub, core, tail_note=core_definitions_md()))
    with open(os.path.join(outdir, f"{split}_structural.md"), "w") as f:
        f.write(render_table_md(hdr+" — Structural", sub_struct, structural))
    if include_alignment and alignment:
        with open(os.path.join(outdir, f"{split}_alignment.md"), "w") as f:
            f.write(render_table_md(hdr+" — Alignment (Appendix)", sub, alignment))
    if include_timing and timing:
        with open(os.path.join(outdir, f"{split}_timing.md"), "w") as f:
            f.write(render_table_md(hdr+" — Timing (Appendix)", sub, timing)) 