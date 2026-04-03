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

def extract_metrics(data, condition_type, sampler_type):
    results = []
    
    for mask_noise, metrics in data[condition_type][sampler_type].items():
        mask_val = float(mask_noise.split('_')[0].replace('mask', ''))
        noise_val = float(mask_noise.split('_')[1].replace('noise', ''))
        
        # Extract main metrics
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
    
    return pd.DataFrame(results)

def create_comparison_table(data):
    # Create tables for both conditional and unconditional results
    cond_random = extract_metrics(data, 'cond', 'random')
    cond_sampler = extract_metrics(data, 'cond', 'sampler')
    uncond_random = extract_metrics(data, 'uncond', 'random')
    uncond_sampler = extract_metrics(data, 'uncond', 'sampler')
    
    # Sort by mask and noise values
    for df in [cond_random, cond_sampler, uncond_random, uncond_sampler]:
        df.sort_values(['Mask', 'Noise'], inplace=True)
    
    # Format the metrics
    def format_metrics(df):
        return df.round(3)
    
    cond_random = format_metrics(cond_random)
    cond_sampler = format_metrics(cond_sampler)
    uncond_random = format_metrics(uncond_random)
    uncond_sampler = format_metrics(uncond_sampler)
    
    # Print tables
    print("Conditional Results:")
    print("\nRandom Sampling:")
    print(cond_random.to_string(index=False))
    
    print("\nSampler:")
    print(cond_sampler.to_string(index=False))
    
    print("\nUnconditional Results:")
    print("\nRandom Sampling:")
    print(uncond_random.to_string(index=False))
    
    print("\nSampler:")
    print(uncond_sampler.to_string(index=False))
    
    # Save to CSV files
    output_dir = os.path.join(WORK_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    cond_random.to_csv(os.path.join(output_dir, 'conditional_random_results.csv'), index=False)
    cond_sampler.to_csv(os.path.join(output_dir, 'conditional_sampler_results.csv'), index=False)
    uncond_random.to_csv(os.path.join(output_dir, 'unconditional_random_results.csv'), index=False)
    uncond_sampler.to_csv(os.path.join(output_dir, 'unconditional_sampler_results.csv'), index=False)
    
    print(f"\nResults have been saved to {output_dir}")

def main():
    json_path = os.environ.get(
        "FPS_SMC_SUMMARY_JSON",
        os.path.join(WORK_DIR, "all_result.json"),
    )
    data = load_json(json_path)
    create_comparison_table(data)

if __name__ == "__main__":
    main() 
