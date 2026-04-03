import json
import os

import numpy as np
import pandas as pd

WORK_DIR = os.environ.get(
    "FPS_SMC_WORK_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_latex_table(df, caption, label):
    # Start table
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "Mask & Noise & PSNR & SSIM & LPIPS & NMSE & RMSE \\\\"
        "\\midrule"
    ]
    
    # Add data rows
    for _, row in df.iterrows():
        latex.append(f"{row['Mask']:.2f} & {row['Noise']:.2f} & {row['PSNR']:.2f} & {row['SSIM']:.3f} & {row['LPIPS']:.3f} & {row['NMSE']:.3f} & {row['RMSE']:.3f} \\\\")
    
    # End table
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

def generate_latex_tables(data):
    # Create tables for both conditional and unconditional results
    tables = {}
    for cond_type in ['cond', 'uncond']:
        for sampler_type in ['random', 'sampler']:
            results = []
            for mask_noise, metrics in data[cond_type][sampler_type].items():
                mask_val = float(mask_noise.split('_')[0].replace('mask', ''))
                noise_val = float(mask_noise.split('_')[1].replace('noise', ''))
                
                row = {
                    'Mask': mask_val,
                    'Noise': noise_val,
                    'PSNR': metrics['psnr'],
                    'SSIM': metrics['ssim'],
                    'LPIPS': metrics['lpips'],
                    'NMSE': metrics['nmse'],
                    'RMSE': metrics['rmse'],
                }
                results.append(row)
            
            df = pd.DataFrame(results)
            df.sort_values(['Mask', 'Noise'], inplace=True)
            
            caption = f"{'Conditional' if cond_type == 'cond' else 'Unconditional'} results with {'random sampling' if sampler_type == 'random' else 'sampler'}"
            label = f"tab:{cond_type}_{sampler_type}"
            
            latex_table = create_latex_table(df, caption, label)
            tables[f"{cond_type}_{sampler_type}"] = latex_table
    
    # Save to file
    output_path = os.path.join(WORK_DIR, 'latex_tables.tex')
    with open(output_path, 'w') as f:
        f.write("% Required packages in preamble:\n")
        f.write("% \\usepackage{booktabs}\n\n")
        for table in tables.values():
            f.write(table)
            f.write("\n\n")
    
    print(f"LaTeX tables have been saved to {output_path}")

def main():
    json_path = os.environ.get(
        "FPS_SMC_SUMMARY_JSON",
        os.path.join(WORK_DIR, "all_result.json"),
    )
    data = load_json(json_path)
    generate_latex_tables(data)

if __name__ == "__main__":
    main() 
