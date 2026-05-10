import pandas as pd
import numpy as np

def generate_latex_table():
    df = pd.read_csv('exp/results/sensitivity.csv')
    
    # We want to pivot the table to show worst_acc for (lambda_inv, lambda_ent)
    # Filter for rotated_mnist and ipdr
    df = df[(df['dataset'] == 'rotated_mnist') & (df['method'] == 'ipdr')]
    
    # Pivot
    pivot_table = df.pivot(index='lambda_inv', columns='lambda_ent', values='worst_acc')
    
    # Format
    latex_str = "\\begin{table}[h]\n"
    latex_str += "  \\centering\n"
    latex_str += "  \\caption{Sensitivity of Worst-Group Accuracy on Rotated MNIST to Hyperparameters $\\lambda_{\\text{inv}}$ and $\\lambda_{\\text{ent}}$.}\n"
    latex_str += "  \\label{tab:sensitivity_table}\n"
    latex_str += "  \\begin{tabular}{l" + "c" * len(pivot_table.columns) + "}\n"
    latex_str += "    \\toprule\n"
    latex_str += "    & \\multicolumn{" + str(len(pivot_table.columns)) + "}{c}{$\\lambda_{\\text{ent}}$} \\\\\n"
    latex_str += "    \\cmidrule(lr){2-" + str(len(pivot_table.columns)+1) + "}\n"
    latex_str += "    $\\lambda_{\\text{inv}}$ & " + " & ".join([str(c) for c in pivot_table.columns]) + " \\\\\n"
    latex_str += "    \\midrule\n"
    
    for idx, row in pivot_table.iterrows():
        row_str = f"    {idx} & " + " & ".join([f"{val:.3f}" if not pd.isna(val) else "-" for val in row]) + " \\\\\n"
        latex_str += row_str
        
    latex_str += "    \\bottomrule\n"
    latex_str += "  \\end{tabular}\n"
    latex_str += "\\end{table}\n"
    
    print(latex_str)

if __name__ == "__main__":
    generate_latex_table()
