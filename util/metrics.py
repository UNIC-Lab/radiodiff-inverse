import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS

def is_building_pixel(pixel, threshold=0.2):
    """判断像素是否属于建筑物（黑色）
    Args:
        pixel: RGB像素值，shape [H,W,C]，范围[0,1]
        threshold: 判断为黑色的阈值
    Returns:
        torch.Tensor: shape [H,W]，1表示建筑物，0表示非建筑物
    """
    # 如果RGB三个通道都小于阈值，认为是黑色（建筑物）
    return torch.all(pixel < threshold, dim=-1)  # 在最后一个维度（通道维度）上判断

def create_building_mask(img):
    """根据图像颜色创建建筑物掩码
    Args:
        img: 图像tensor，shape [C,H,W] 或 [B,C,H,W]，范围[0,1]
    Returns:
        building_mask: 建筑物掩码，shape [1,1,H,W]，1表示建筑物，0表示非建筑物
    """
    if img.dim() == 4:  # [B,C,H,W]
        img = img[0]  # 取第一个batch
    
    # 转换为[H,W,C]便于处理
    img = img.permute(1, 2, 0)
    
    # 创建掩码 [H,W]
    building_mask = is_building_pixel(img)
    
    # 转换回[1,1,H,W]格式
    building_mask = building_mask.unsqueeze(0).unsqueeze(0)
    
    return building_mask.float()

def calculate_nmse(original, generated):
    """计算归一化均方误差"""
    return torch.mean((original - generated) ** 2) / torch.mean(original ** 2)

def calculate_rmse(original, generated):
    """计算均方根误差"""
    return torch.sqrt(torch.mean((original - generated) ** 2))

def calculate_ssim(original, generated):
    """计算结构相似性"""
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(generated):
        generated = generated.cpu().numpy()
    
    # 确保数据在[0,1]范围内
    original = (original - original.min()) / (original.max() - original.min())
    generated = (generated - generated.min()) / (generated.max() - generated.min())
    
    # 处理维度
    if original.ndim == 4:  # batch, channel, height, width
        original = original[0]  # 取第一个batch
        generated = generated[0]
    
    # 转换为正确的格式：(height, width, channel)
    if original.ndim == 3 and original.shape[0] == 3:  # channel, height, width
        original = np.transpose(original, (1, 2, 0))
        generated = np.transpose(generated, (1, 2, 0))
    
    return ssim(original, generated, data_range=1.0, channel_axis=2)  # 指定channel_axis

def calculate_psnr(original, generated):
    """计算峰值信噪比"""
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(generated):
        generated = generated.cpu().numpy()
    
    # 确保数据在[0,1]范围内
    original = (original - original.min()) / (original.max() - original.min())
    generated = (generated - generated.min()) / (generated.max() - generated.min())
    
    return psnr(original, generated, data_range=1.0)

def find_brightest_point(img):
    """找到图像中最亮的点
    Args:
        img: 图像tensor，shape [C,H,W] 或 [B,C,H,W]，范围[0,1]
    Returns:
        tuple: (y, x) 坐标
    """
    if img.dim() == 4:
        img = img[0]  # 取第一个batch
    
    # 计算亮度（RGB平均值）
    brightness = torch.mean(img, dim=0)  # [H,W]
    
    # 找到最亮点的坐标
    max_idx = torch.argmax(brightness)
    h, w = brightness.shape
    y = max_idx // w  # 行号
    x = max_idx % w   # 列号
    
    return y.item(), x.item()

def calculate_source_error(recon_img, gt_img):
    """计算信号源位置误差
    Args:
        recon_img: 重建图像
        gt_img: 原始图像
    Returns:
        float: 欧氏距离
    """
    # 找到两个图像中的最亮点
    y1, x1 = find_brightest_point(recon_img)
    y2, x2 = find_brightest_point(gt_img)
    
    # 计算欧氏距离
    distance = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
    
    return distance

def preprocess_for_metrics(img):
    """预处理图像用于计算指标
    Args:
        img: PyTorch tensor, shape [B,C,H,W] 或 [C,H,W]
    Returns:
        numpy array, shape [H,W,C], 范围[0,1]
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    # 处理batch维度
    if img.ndim == 4:  # [B,C,H,W] -> [C,H,W]
        img = img[0]
    
    # 转换为[H,W,C]格式
    if img.shape[0] == 3 and img.ndim == 3:  # 如果是[C,H,W]格式
        img = np.transpose(img, (1, 2, 0))
    
    # 确保数据在[0,1]范围内
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    return img

def evaluate_metrics(recon_img, gt_img, degraded_img, device='cuda', lpips_fn=None):
    """评估重建质量，包括整体、建筑物区域和光源区域"""
    # 确保输入是4D张量 [B,C,H,W]
    if recon_img.dim() == 3:
        recon_img = recon_img.unsqueeze(0)
    if gt_img.dim() == 3:
        gt_img = gt_img.unsqueeze(0)
    
    metrics = {}
    
    # 1. 整体图像质量指标
    recon_np = preprocess_for_metrics(recon_img)
    gt_np = preprocess_for_metrics(gt_img)
    
    metrics['psnr'] = psnr(gt_np, recon_np, data_range=1.0)
    metrics['ssim'] = ssim(gt_np, recon_np, data_range=1.0, channel_axis=2)
    metrics['nmse'] = calculate_nmse(gt_img, recon_img).item()
    metrics['rmse'] = calculate_rmse(gt_img, recon_img).item()
    
    if lpips_fn is None:
        # 确保LPIPS模型在正确的设备上
        lpips_fn = LPIPS(net='alex').to(recon_img.device)
    metrics['lpips'] = lpips_fn(recon_img * 2 - 1, gt_img * 2 - 1).item()
    
    # 2. 光源区域指标
    metrics['source_error'] = calculate_source_error(recon_img, gt_img)
    
    source_y, source_x = find_brightest_point(gt_img)
    h, w = gt_img.shape[-2:]
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
    y_grid = y_grid.to(recon_img.device)
    x_grid = x_grid.to(recon_img.device)
    
    sigma = 20.0
    source_weight = torch.exp(-((y_grid - source_y)**2 + (x_grid - source_x)**2) / (2 * sigma**2))
    source_weight = source_weight.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]
    
    source_recon = recon_img * source_weight
    source_gt = gt_img * source_weight
    
    source_recon_np = preprocess_for_metrics(source_recon)
    source_gt_np = preprocess_for_metrics(source_gt)
    
    metrics['source_psnr'] = psnr(source_gt_np, source_recon_np, data_range=1.0)
    metrics['source_ssim'] = ssim(source_gt_np, source_recon_np, data_range=1.0, channel_axis=2)
    metrics['source_nmse'] = calculate_nmse(source_gt, source_recon).item()
    metrics['source_rmse'] = calculate_rmse(source_gt, source_recon).item()
    metrics['source_lpips'] = lpips_fn(source_recon * 2 - 1, source_gt * 2 - 1).item()
    
    # 3. 建筑物区域指标
    building_mask = create_building_mask(gt_img)  # [1,1,H,W]
    
    # 扩展掩码到3个通道
    building_mask = building_mask.repeat(1, 3, 1, 1)  # [1,3,H,W]
    
    building_recon = recon_img * building_mask
    building_gt = gt_img * building_mask
    
    building_recon_np = preprocess_for_metrics(building_recon)
    building_gt_np = preprocess_for_metrics(building_gt)
    
    metrics['building_psnr'] = psnr(building_gt_np, building_recon_np, data_range=1.0)
    metrics['building_ssim'] = ssim(building_gt_np, building_recon_np, data_range=1.0, channel_axis=2)
    metrics['building_nmse'] = calculate_nmse(building_gt, building_recon).item()
    metrics['building_rmse'] = calculate_rmse(building_gt, building_recon).item()
    metrics['building_lpips'] = lpips_fn(building_recon * 2 - 1, building_gt * 2 - 1).item()
    
    return metrics 