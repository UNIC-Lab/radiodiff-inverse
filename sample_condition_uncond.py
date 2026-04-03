import os
import json
from functools import partial
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from util.metrics import evaluate_metrics
from util.experiment_utils import create_experiment_dir, save_experiment_results

def create_progression_grid(images_dir, output_path, grid_size=None):
    """将中间结果图像合成为一张大图
    
    Args:
        images_dir: 包含x_XXXX.png格式图像的目录
        output_path: 输出的大图路径
        grid_size: (行数,列数)的元组，如果为None则自动计算
    """
    # 获取所有x_开头的图像并排序
    image_files = [f for f in os.listdir(images_dir) if f.startswith('x_') and f.endswith('.png')]
    image_files.sort()
    
    if not image_files:
        print(f"没有找到中间结果图像在 {images_dir}")
        return
    
    # 读取第一张图像以获取尺寸
    first_img = Image.open(os.path.join(images_dir, image_files[0]))
    img_width, img_height = first_img.size
    
    # 确定网格大小
    num_images = len(image_files)
    if grid_size is None:
        # 自动计算网格大小，尽量接近正方形
        grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)
    else:
        grid_rows, grid_cols = grid_size
    
    # 创建大图画布
    grid_width = grid_cols * img_width
    grid_height = grid_rows * img_height
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 放置图像
    for idx, img_file in enumerate(image_files):
        if idx >= grid_rows * grid_cols:
            break
            
        img = Image.open(os.path.join(images_dir, img_file))
        row = idx // grid_cols
        col = idx % grid_cols
        grid_img.paste(img, (col * img_width, row * img_height))
    
    # 保存大图
    grid_img.save(output_path)
    print(f"已生成进度网格图：{output_path}")
    
    return output_path

def run_experiment(args):
    """运行无条件掩码实验"""
    # 加载配置
    model_config = yaml.safe_load(open(args.model_config))
    diffusion_config = yaml.safe_load(open(args.diffusion_config))
    task_config = yaml.safe_load(open(args.task_config))
    
    # 设备配置
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    
    # 准备数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((256, 256)),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_dataset(**task_config['data'], transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    
    # 加载建筑物掩码（仅用于评估）
    building_dataset = get_dataset(
        name='building',
        root=task_config['data']['root'],
        image_dir=task_config['data']['image_dir'],
        building_dir=task_config['data']['building_dir'],
        transforms=transform
    )
    building_loader = get_dataloader(building_dataset, batch_size=1, num_workers=0, train=False)
    
    # 实验结果存储
    all_results = {}
    
    # 遍历所有参数组合
    for mask_ratio in args.mask_ratios:
        for noise_level in args.noise_levels:
            # 更新配置
            task_config['measurement']['mask_opt']['mask_ratio'] = mask_ratio
            task_config['measurement']['noise']['sigma'] = noise_level
            
            # 创建实验目录，使用新的目录结构
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
            
            # 运行实验
            metrics_list = []
            for i, ((img, building_mask, img_path), (_, building_mask_2, _)) in enumerate(zip(loader, building_loader)):
                if i >= args.num_images:
                    break
                
                img = img.to(device)
                building_mask = building_mask.to(device)
                
                # 生成掩码和测量值（不使用建筑物信息）
                mask = mask_generator(**task_config['measurement']['mask_opt'])(img)
                y = operator.forward(img, mask=mask)
                y_n = noiser(y)
                
                # 确保存在progress目录
                if args.record:
                    progress_dir = os.path.join(exp_dir, f'progress_{i:04d}')
                    os.makedirs(progress_dir, exist_ok=True)
                
                # 采样（不使用建筑物信息）
                x_start = torch.randn_like(img)
                sample, _ = sample_fn(
                    x_start=x_start,
                    measurement=y_n,
                    operator=operator,
                    op='inpainting',
                    mask=mask,
                    record=args.record,
                    save_root=progress_dir if args.record else exp_dir
                )
                
                # 评估指标（包括建筑物区域的评估）
                metrics = evaluate_metrics(
                    sample, 
                    img, 
                    operator.forward(img, mask=mask),
                    device=device
                )
                metrics_list.append(metrics)
                
                # 保存结果
                os.makedirs(os.path.join(exp_dir, 'input'), exist_ok=True)
                os.makedirs(os.path.join(exp_dir, 'label'), exist_ok=True)
                os.makedirs(os.path.join(exp_dir, 'recon'), exist_ok=True)
                os.makedirs(os.path.join(exp_dir, 'building_mask'), exist_ok=True)
                
                plt.imsave(os.path.join(exp_dir, 'input', f"{i:04d}_input.png"), clear_color(y_n))
                plt.imsave(os.path.join(exp_dir, 'label', f"{i:04d}_label.png"), clear_color(img))
                plt.imsave(os.path.join(exp_dir, 'recon', f"{i:04d}_recon.png"), clear_color(sample))
                plt.imsave(os.path.join(exp_dir, 'building_mask', f"{i:04d}_mask.png"), clear_color(building_mask))
                
                # 如果记录中间结果，则创建进度网格图
                if args.record:
                    grid_path = os.path.join(exp_dir, f"progression_grid_{i:04d}.png")
                    create_progression_grid(progress_dir, grid_path)
                    
                    # 同样为测量值y创建网格图
                    y_progress_dir = os.path.join(progress_dir)
                    y_grid_path = os.path.join(exp_dir, f"y_progression_grid_{i:04d}.png")
                    y_image_files = [f for f in os.listdir(y_progress_dir) if f.startswith('y_') and f.endswith('.png')]
                    if y_image_files:
                        create_progression_grid(y_progress_dir, y_grid_path)
            
            # 计算平均指标
            avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
            
            # 保存结果
            exp_key = f"uncond_mask{mask_ratio:.2f}_noise{noise_level:.3f}"
            all_results[exp_key] = avg_metrics
            
            # 保存当前实验的指标
            with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                json.dump(avg_metrics, f, indent=4)
    
    # 保存所有实验结果
    with open(os.path.join(args.save_dir, 'uncond_all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_imagenet_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/inpainting_config.yaml')
    parser.add_argument('--save_dir', type=str, default='results/default_uncond')
    parser.add_argument('--c_rate', type=float, default=0.95)
    parser.add_argument('--particle_size', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--mask_ratios', type=float, nargs='+', 
                       default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.05, 0.1, 0.15])
    parser.add_argument('--num_images', type=int, default=200)
    parser.add_argument('--version', type=str, default='v2')
    parser.add_argument('--record', action='store_true', help='记录重建过程中的中间结果')
    parser.add_argument('--grid_rows', type=int, default=None, help='进度网格图的行数')
    parser.add_argument('--grid_cols', type=int, default=None, help='进度网格图的列数')
    
    args = parser.parse_args()
    run_experiment(args) 
