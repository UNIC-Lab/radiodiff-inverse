import os
import json

from functools import partial
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from piq import psnr, ssim
from piq.perceptual import LPIPS

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.svd_replacement import Deblurring, Deblurring2D
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, Blurkernel
from util.logger import get_logger
from util.metrics import evaluate_metrics
from util.experiment_utils import create_experiment_dir, save_experiment_results


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def run_experiment(args):
    """运行批量实验"""
    # 加载配置
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # 设备配置
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    # 加载模型
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    
    # 准备数据加载器
    data_config = task_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((256, 256)),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    
    # 实验结果存储
    all_results = {}
    
    # 遍历所有参数组合
    for mask_ratio in args.mask_ratios:
        for noise_level in args.noise_levels:
            # 更新配置
            task_config['measurement']['mask_opt']['mask_ratio'] = mask_ratio
            task_config['measurement']['noise']['sigma'] = noise_level
            
            # 创建实验目录
            exp_dir = create_experiment_dir(
                base_dir=args.save_dir,
                mask_type=task_config['measurement']['mask_opt']['mask_type'],
                mask_ratio=mask_ratio,
                noise_level=noise_level,
                version=args.version
            )
            
            # 准备操作器和噪声
            operator = get_operator(device=device, **task_config['measurement']['operator'])
            noiser = get_noise(**task_config['measurement']['noise'])
            
            # 准备条件方法
            cond_method = get_conditioning_method(
                task_config['conditioning']['method'],
                operator,
                noiser,
                **task_config['conditioning']['params']
            )
            
            # 准备采样器
            sampler = create_sampler(**diffusion_config, c_rate=args.c_rate, particle_size=args.particle_size)
            sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=cond_method.conditioning)
            
            # 运行实验并收集指标
            metrics_list = []
            for i, (ref_img, building_mask, img_path) in enumerate(loader):
                if i >= args.num_images:
                    break
                
                # 从原始图像路径提取文件名
                filename = os.path.basename(img_path[0])  # [0]因为batch_size=1
                
                ref_img = ref_img.to(device)
                building_mask = building_mask.to(device)
                
                # 生成掩码和测量值
                mask = mask_generator(**task_config['measurement']['mask_opt'])(ref_img, building_mask)
                y = operator.forward(ref_img, mask=mask)
                y_n = noiser(y)
                
                # 采样
                x_start = torch.randn_like(ref_img).requires_grad_()
                sample, _ = sample_fn(
                    x_start=x_start,
                    measurement=y_n,
                    operator=operator,
                    op='inpainting',
                    mask=mask,
                    record=False,
                    save_root=exp_dir
                )
                
                # 保存图像，使用原始文件名
                plt.imsave(os.path.join(exp_dir, 'input', filename), clear_color(y_n))
                plt.imsave(os.path.join(exp_dir, 'label', filename), clear_color(ref_img))
                plt.imsave(os.path.join(exp_dir, 'recon', filename), clear_color(sample))
                
                # 计算指标
                metrics = evaluate_metrics(ref_img, sample, operator.forward(ref_img, mask=mask))
                metrics_list.append(metrics)
            
            # 计算平均指标
            avg_metrics = {}
            for key in metrics_list[0].keys():
                # 将 numpy float32 转换为 Python float
                avg_metrics[key] = float(np.mean([float(m[key]) for m in metrics_list]))
            
            # 保存结果
            exp_key = f"mask{mask_ratio:.2f}_noise{noise_level:.3f}"
            all_results[exp_key] = avg_metrics
            
            # 保存当前实验的指标
            with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                json.dump(avg_metrics, f, indent=4)
    
    # 保存所有实验结果
    save_experiment_results(exp_dir, all_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_imagenet_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/inpainting_config_imagenet.yaml')
    parser.add_argument('--save_dir', type=str, default='results/default_cond')
    parser.add_argument('--c_rate', type=float, default=0.95)
    parser.add_argument('--particle_size', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--mask_ratios', type=float, nargs='+', 
                       default=[0.95],
                       help='Mask coverage ratios to test')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.05])
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to process')
    parser.add_argument('--version', type=str, default='v1')
    
    args = parser.parse_args()
    run_experiment(args)
