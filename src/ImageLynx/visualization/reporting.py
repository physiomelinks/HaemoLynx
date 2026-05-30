import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use Agg backend to avoid GUI issues in CI/headless environments
matplotlib.use('Agg')

def generate_model_results_dashboard(vtk_export: dict, perfusion_field: dict, output_dir: Path):
    """
    Extracts 1D network metrics and 3D perfusion metrics, generates static 
    distribution plots (Histogram + KDE), and writes a Markdown dashboard.
    """
    # 1. Define Target Metrics
    metrics_1d = [
        "assigned_diameter_um", "flow_abs", "flow_signed", "hematocrit", 
        "pressure_drop", "resistance", "viscosity", "wall_shear_stress_pa"
    ]
    metrics_3d = ["PO2_mmhg", "PCO2_mmhg", "pH"]
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    markdown_path = output_dir / "model_results.md"
    
    md_content = [
        "# Model Results Dashboard\n",
        "This dashboard provides distribution analysis of the key physical and hemodynamic outputs from the pipeline.\n"
    ]
    
    # 2. Extract and Plot 1D Metrics
    md_content.append("## 1D Hemodynamic Network Metrics\n")
    cell_data = vtk_export.get("cell_data", {})
    
    for metric in metrics_1d:
        if metric in cell_data:
            data = np.asarray(cell_data[metric], dtype=float)
            data = data[~np.isnan(data)] # Drop NaNs
            if len(data) > 0:
                _create_distribution_plot(data, metric, plots_dir)
                _append_markdown_section(md_content, data, metric)
            else:
                md_content.append(f"### {metric}\n*Data array was empty or contained only NaNs.*\n")
        else:
            md_content.append(f"### {metric}\n*Metric not found in pipeline output.*\n")
            
    # 3. Extract and Plot 3D Metrics
    md_content.append("## 3D Tissue Perfusion Metrics\n")
    for metric in metrics_3d:
        if perfusion_field and metric in perfusion_field:
            data = np.asarray(perfusion_field[metric], dtype=float)
            # Flatten and drop NaNs or zeros if the space is unperfused background
            data = data.flatten()
            data = data[~np.isnan(data)]
            
            # For 3D fields, we often mask out pure zero background if it represents "outside tissue"
            # But PO2 can physically be near zero. Let's just plot valid numbers.
            if len(data) > 0:
                _create_distribution_plot(data, metric, plots_dir)
                _append_markdown_section(md_content, data, metric)
            else:
                md_content.append(f"### {metric}\n*Data array was empty or contained only NaNs.*\n")
        else:
            md_content.append(f"### {metric}\n*Metric not found in pipeline output (solver may be disabled).*\n")
            
    # 4. Write Markdown File (Overwrite mode)
    with open(markdown_path, "w") as f:
        f.write("\n".join(md_content))
        
    print(f"\nDashboard generated: {markdown_path}")

def _create_distribution_plot(data: np.ndarray, metric_name: str, plots_dir: Path):
    """Generates a Seaborn distribution plot and saves it as a PNG, aggressively overwriting."""
    plt.figure(figsize=(8, 5))
    
    # Handle the case where all values are identical (kdeplot crashes on zero variance)
    if np.std(data) < 1e-8:
        sns.histplot(data, bins=10, color="dodgerblue")
    else:
        sns.histplot(data, kde=True, bins=30, color="dodgerblue", line_kws={'linewidth': 2})
        
    plt.title(f"Distribution of {metric_name}", fontsize=14, pad=10)
    plt.xlabel(metric_name, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = plots_dir / f"{metric_name}_dist.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

def _append_markdown_section(md_content: list, data: np.ndarray, metric_name: str):
    """Calculates statistics and appends the Markdown formatting."""
    mean_val = np.mean(data)
    median_val = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    md_content.append(f"### {metric_name}")
    md_content.append(f"**Mean:** `{mean_val:.4f}` | **Median:** `{median_val:.4f}` | **Min:** `{min_val:.4f}` | **Max:** `{max_val:.4f}`\n")
    md_content.append(f"![{metric_name} Distribution](plots/{metric_name}_dist.png)\n")
    md_content.append("---\n")
