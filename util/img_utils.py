import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from motionblur.motionblur import Kernel
from .fastmri_utils import fft2c_new, ifft2c_new


"""
Helper functions for new types of inverse problems
"""

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)


def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    
    # 检查维度并相应处理
    if x.ndim == 2:  # 单通道灰度图
        return normalize_np(x)
    elif x.ndim == 3:  # RGB图像或带batch的灰度图
        if x.shape[0] == 1:  # 单通道
            return normalize_np(x.squeeze())
        else:  # RGB
            return normalize_np(np.transpose(x, (1, 2, 0)))
    else:
        raise ValueError(f"Unexpected input dimension: {x.ndim}")


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


def prepare_im(load_dir, image_size, device):
    ref_img = torch.from_numpy(normalize_np(plt.imread(load_dir)[:, :, :3].astype(np.float32))).to(device)
    ref_img = ref_img.permute(2, 0, 1)
    ref_img = ref_img.view(1, 3, image_size, image_size)
    ref_img = ref_img * 2 - 1
    return ref_img


def fold_unfold(img_t, kernel, stride):
    img_shape = img_t.shape
    B, C, H, W = img_shape
    print("\n----- input shape: ", img_shape)

    patches = img_t.unfold(3, kernel, stride).unfold(2, kernel, stride).permute(0, 1, 2, 3, 5, 4)

    print("\n----- patches shape:", patches.shape)
    # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C, -1, kernel*kernel)
    print("\n", patches.shape) # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)
    print("\n", patches.shape) # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(B, C*kernel*kernel, -1)
    print("\n", patches.shape) # [B, C*prod(kernel_size), L] as expected by Fold

    output = F.fold(patches, output_size=(H, W),
                    kernel_size=kernel, stride=stride)
    # mask that mimics the original folding:
    recovery_mask = F.fold(torch.ones_like(patches), output_size=(
        H, W), kernel_size=kernel, stride=stride)
    output = output/recovery_mask

    return patches, output


def reshape_patch(x, crop_size=128, dim_size=3):
    x = x.transpose(0, 2).squeeze()  # [9, 3*(128**2)]
    x = x.view(dim_size**2, 3, crop_size, crop_size)
    return x

def reshape_patch_back(x, crop_size=128, dim_size=3):
    x = x.view(dim_size**2, 3*(crop_size**2)).unsqueeze(dim=-1)
    x = x.transpose(0, 2)
    return x


class Unfolder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.unfold = nn.Unfold(crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, x):
        patch1D = self.unfold(x)
        patch2D = reshape_patch(patch1D, crop_size=self.crop_size, dim_size=self.dim_size)
        return patch2D


def center_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

class Folder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.fold = nn.Fold(img_size, crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, patch2D):
        patch1D = reshape_patch_back(patch2D, crop_size=self.crop_size, dim_size=self.dim_size)
        return self.fold(patch1D)


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w

class mask_generator:
    def __init__(self, mask_type='fixed_rect', mask_ratio=0.5, image_size=256, margin=(0, 0), box_size=(64, 64)):
        """
        Args:
            mask_type: 掩码类型，支持 'fixed_rect', 'rect_sample', 'sensor_rect', 
                              'sensor_rect_uncond', 'random_cond', 'random_uncond', 
                              'box_random', 'corner_box_random'
            mask_ratio: 目标掩码覆盖率（0到1之间）
            image_size: 图像大小
            margin: 边缘留白
            box_size: box_random模式下使用的大方块尺寸，格式为(height, width)
        """
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.margin = margin
        self.box_size = box_size
        if hasattr(self, 'mask_len_range'):
            self.box_h, self.box_w = self.mask_len_range

    def _retrieve_box(self, img):
        """Generate a single random box mask"""
        l, h = self.mask_len_range
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, _, _, _, _ = random_sq_bbox(
            img, mask_shape=(mask_h, mask_w),
            image_size=self.image_size, margin=self.margin
        )
        return mask

    def _retrieve_random(self, img):
        """Generate random pixel-wise mask"""
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(
            self.image_size * self.image_size,
            int(self.image_size * self.image_size * prob),
            replace=False
        )
        mask_vec[:, samples] = 0
        mask = mask_vec.view(1, self.image_size, self.image_size).repeat(3, 1, 1)
        return torch.ones_like(img, device=img.device) * mask

    def _retrieve_fixed_rect(self, img, building_mask=None):
        """生成不重叠的2x3矩形掩码，精确控制覆盖率
        掩码值为0表示被遮挡区域，值为1表示保留区域
        Args:
            img: 输入图像 [B, C, H, W]
            building_mask: 建筑物掩码，与img相同大小。1表示建筑物位置，0表示非建筑物位置
        """
        B, C, H, W = img.shape
        mask = torch.ones_like(img, device=img.device)
        
        # 计算有效区域边界
        max_y = self.image_size - self.margin[0] - 2
        max_x = self.image_size - self.margin[1] - 3
        
        if max_y <= self.margin[0] or max_x <= self.margin[1]:
            raise ValueError("Margin too large for 2x3 masks")

        # 创建掩码位置记录数组
        mask_positions = torch.zeros((H, W), device=img.device)
        
        # 计算非建筑区域和目标掩码面积
        if building_mask is not None:
            # 计算建筑物区域（只需要一个通道的掩码）
            building_area = torch.sum(building_mask[0, 0])
            non_building_area = H * W - building_area
            
            # 计算需要掩码的像素数（非建筑区域的mask_ratio比例）
            target_pixels = int(non_building_area * self.mask_ratio)
        else:
            # 如果没有建筑物掩码，使用整个图像面积
            target_pixels = int(H * W * self.mask_ratio)
        
        # 计算需要的2x3掩码数量
        num_masks_needed = target_pixels // 6  # 每个2x3掩码覆盖6个像素

        successful_masks = 0
        max_attempts = num_masks_needed * 10
        attempts = 0

        while successful_masks < num_masks_needed and attempts < max_attempts:
            y = torch.randint(low=self.margin[0], high=max_y, size=(1,)).item()
            x = torch.randint(low=self.margin[1], high=max_x, size=(1,)).item()
            
            # 检查是否有重叠
            if mask_positions[y:y+2, x:x+3].sum() > 0:
                attempts += 1
                continue
            
            # 检查是否与建筑物重叠
            if building_mask is not None:
                region = building_mask[0, 0, y:y+2, x:x+3]
                if region.sum() > 0:  # 如果与建筑物有任何重叠，跳过
                    attempts += 1
                    continue
            
            # 应用掩码
            mask[:, :, y:y+2, x:x+3] = 0
            mask_positions[y:y+2, x:x+3] = 1
            successful_masks += 1
            attempts += 1

        # 打印统计信息
        total_pixels = H * W
        masked_pixels = total_pixels - torch.sum(mask[0, 0])  # 只计算一个通道
        
        print(f"Total image pixels: {total_pixels}")
        if building_mask is not None:
            print(f"Building pixels: {building_area}")
            print(f"Non-building pixels: {non_building_area}")
            print(f"Target masked pixels: {target_pixels}")
            print(f"Actual masked pixels: {masked_pixels}")
            print(f"Non-building masked ratio: {(masked_pixels / non_building_area).item():.2%}")
        else:
            print(f"Target masked pixels: {target_pixels}")
            print(f"Actual masked pixels: {masked_pixels}")
            print(f"Masked ratio: {(masked_pixels / total_pixels).item():.2%}")
        
        if successful_masks < num_masks_needed:
            print(f"Warning: Only placed {successful_masks}/{num_masks_needed} masks after {attempts} attempts")
        
        return mask

    def _retrieve_rect_sample(self, img, building_mask=None):
        """生成不重叠的2x3矩形采样点，精确控制采样率
        与fixed_rect相反，此方法中掩码值为1的地方是采样点（保留信息的区域），值为0的地方是被mask的区域
        Args:
            img: 输入图像 [B, C, H, W]
            building_mask: 建筑物掩码，与img相同大小。1表示建筑物位置，0表示非建筑物位置
        """
        B, C, H, W = img.shape
        mask = torch.zeros_like(img, device=img.device)  # 初始化为全0（表示遮盖所有区域）
        
        # 计算有效区域边界
        max_y = self.image_size - self.margin[0] - 2
        max_x = self.image_size - self.margin[1] - 3
        
        if max_y <= self.margin[0] or max_x <= self.margin[1]:
            raise ValueError("Margin too large for 2x3 sampling rectangles")

        # 创建采样位置记录数组
        sample_positions = torch.zeros((H, W), device=img.device)
        
        # 计算非建筑区域和目标采样像素数
        if building_mask is not None:
            # 计算建筑物区域（只需要一个通道的掩码）
            building_area = torch.sum(building_mask[0, 0])
            non_building_area = H * W - building_area
            
            # 计算需要采样的像素数（非建筑区域的(1-mask_ratio)比例）
            target_pixels = int(non_building_area * (1 - self.mask_ratio))
        else:
            # 如果没有建筑物掩码，使用整个图像面积
            target_pixels = int(H * W * (1 - self.mask_ratio))
        
        # 计算需要的2x3采样块数量
        num_samples_needed = target_pixels // 6  # 每个2x3采样块包含6个像素
        
        successful_samples = 0
        max_attempts = num_samples_needed * 10
        attempts = 0
        
        while successful_samples < num_samples_needed and attempts < max_attempts:
            y = torch.randint(low=self.margin[0], high=max_y, size=(1,)).item()
            x = torch.randint(low=self.margin[1], high=max_x, size=(1,)).item()
            
            # 检查是否有重叠
            if sample_positions[y:y+2, x:x+3].sum() > 0:
                attempts += 1
                continue
            
            # 检查是否与建筑物重叠（如果有建筑物掩码）
            if building_mask is not None:
                region = building_mask[0, 0, y:y+2, x:x+3]
                if region.sum() > 0:  # 如果与建筑物有任何重叠，跳过
                    attempts += 1
                    continue
            
            # 应用采样点（设置为1，表示这些位置的信息被保留）
            mask[:, :, y:y+2, x:x+3] = 1
            sample_positions[y:y+2, x:x+3] = 1
            successful_samples += 1
            attempts += 1
        
        # 如果有建筑物掩码，也保留建筑物区域
        if building_mask is not None:
            mask = mask + building_mask  # 建筑物区域也设为1
            # 确保值不超过1
            mask = torch.clamp(mask, 0, 1)
        
        # 打印统计信息
        total_pixels = H * W
        sampled_pixels = torch.sum(mask[0, 0])  # 只计算一个通道
        
        print(f"Total image pixels: {total_pixels}")
        if building_mask is not None:
            building_area = torch.sum(building_mask[0, 0])
            non_building_area = H * W - building_area
            print(f"Building pixels: {building_area}")
            print(f"Non-building pixels: {non_building_area}")
            print(f"Target sampled pixels: {target_pixels}")
            print(f"Actual sampled pixels (excluding buildings): {sampled_pixels - building_area}")
            print(f"Non-building sampling ratio: {((sampled_pixels - building_area) / non_building_area).item():.2%}")
            print(f"Overall mask ratio: {(1 - sampled_pixels / total_pixels).item():.2%}")
        else:
            print(f"Target sampled pixels: {target_pixels}")
            print(f"Actual sampled pixels: {sampled_pixels}")
            print(f"Sampling ratio: {(sampled_pixels / total_pixels).item():.2%}")
            print(f"Mask ratio: {(1 - sampled_pixels / total_pixels).item():.2%}")
        
        if successful_samples < num_samples_needed:
            print(f"Warning: Only placed {successful_samples}/{num_samples_needed} sampling blocks after {attempts} attempts")
        
        return mask

    def _retrieve_box_random_mask(self, img, building_mask=None):
        """先对图像的一个区域完全mask，然后在其他区域进行随机像素级mask
        掩码值为0表示被遮挡区域，值为1表示保留区域
        Args:
            img: 输入图像 [B, C, H, W]
            building_mask: 建筑物掩码，与img相同大小。1表示建筑物位置，0表示非建筑物位置
        """
        B, C, H, W = img.shape
        mask = torch.ones_like(img, device=img.device)  # 初始化为全1（表示所有区域都保留）
        
        # 1. 计算总的需要mask的像素数
        if building_mask is not None:
            building_area = torch.sum(building_mask[0, 0])
            non_building_area = H * W - building_area
            target_pixels = int(non_building_area * self.mask_ratio)
        else:
            target_pixels = int(H * W * self.mask_ratio)
            
        # 2. 放置一个大mask区域
        box_h, box_w = self.box_size
        
        # 确保box尺寸不超过图像尺寸
        box_h = min(box_h, H - 2 * self.margin[0])
        box_w = min(box_w, W - 2 * self.margin[1])
        
        # 计算可能的位置范围
        max_y = H - self.margin[0] - box_h
        max_x = W - self.margin[1] - box_w
        
        # 找到一个合适的位置放置大方块
        attempts = 0
        max_attempts = 50
        big_box_placed = False
        
        while not big_box_placed and attempts < max_attempts:
            # 随机选择位置
            t = np.random.randint(self.margin[0], max_y)
            l = np.random.randint(self.margin[1], max_x)
            
            # 检查是否与建筑物重叠
            if building_mask is not None:
                region = building_mask[0, 0, t:t+box_h, l:l+box_w]
                if region.sum() > 0:  # 如果与建筑物有任何重叠，跳过
                    attempts += 1
                    continue
            
            # 放置大方块掩码（设置为0，表示这些位置被遮挡）
            mask[:, :, t:t+box_h, l:l+box_w] = 0
            big_box_placed = True
            break
        
        if not big_box_placed:
            print(f"Warning: Could not place large box mask after {attempts} attempts")
            # 如果无法放置大方块，尝试放置一个较小的方块
            box_h //= 2
            box_w //= 2
            t = np.random.randint(self.margin[0], H - self.margin[0] - box_h)
            l = np.random.randint(self.margin[1], W - self.margin[1] - box_w)
            mask[:, :, t:t+box_h, l:l+box_w] = 0
        
        # 3. 计算大方块已mask的像素数
        box_pixels = box_h * box_w
        remaining_pixels = target_pixels - box_pixels
        
        # 如果需要mask的像素不足，直接返回
        if remaining_pixels <= 0:
            print(f"Box mask pixels: {box_pixels}")
            print(f"Target mask pixels: {target_pixels}")
            print(f"No need for additional random masking")
            return mask
        
        # 4. 创建一个标记已mask区域的掩码
        masked_positions = torch.zeros((H, W), device=img.device)
        masked_positions[t:t+box_h, l:l+box_w] = 1  # 标记大方块区域
        
        if building_mask is not None:
            # 标记建筑物区域为已mask，这样就不会在建筑物上随机mask
            masked_positions = masked_positions + building_mask[0, 0]
            # 确保值不超过1
            masked_positions = torch.clamp(masked_positions, 0, 1)
        
        # 5. 生成随机像素mask
        # 找出所有可以mask的像素位置（未被大方块或建筑物占用的位置）
        available_positions = torch.where(masked_positions == 0)
        available_count = len(available_positions[0])
        
        if available_count <= remaining_pixels:
            # 如果剩余可用像素不足，则全部mask
            mask[:, :, available_positions[0], available_positions[1]] = 0
            print(f"Warning: Not enough available pixels. Masked all {available_count} available pixels.")
        else:
            # 随机选择像素进行mask
            random_indices = np.random.choice(available_count, size=remaining_pixels, replace=False)
            y_indices = available_positions[0][random_indices]
            x_indices = available_positions[1][random_indices]
            mask[:, :, y_indices, x_indices] = 0
        
        # 6. 打印统计信息
        total_pixels = H * W
        masked_pixels = total_pixels - torch.sum(mask[0, 0])  # 只计算一个通道
        
        print(f"Total image pixels: {total_pixels}")
        print(f"Box mask pixels: {box_pixels}")
        print(f"Additional random masked pixels: {masked_pixels - box_pixels}")
        
        if building_mask is not None:
            building_area = torch.sum(building_mask[0, 0])
            non_building_area = H * W - building_area
            print(f"Building pixels: {building_area}")
            print(f"Non-building pixels: {non_building_area}")
            print(f"Target masked pixels: {target_pixels}")
            print(f"Actual masked pixels in non-building areas: {masked_pixels}")
            print(f"Non-building masked ratio: {(masked_pixels / non_building_area).item():.2%}")
        else:
            print(f"Target masked pixels: {target_pixels}")
            print(f"Actual masked pixels: {masked_pixels}")
            print(f"Masked ratio: {(masked_pixels / total_pixels).item():.2%}")
        
        return mask

    def _retrieve_corner_box_random_mask(self, img, building_mask=None):
        """在图像四个角落之一放置四分之一大小的方块掩码，然后在其他区域进行随机像素级mask
        掩码值为0表示被遮挡区域，值为1表示保留区域
        Args:
            img: 输入图像 [B, C, H, W]
            building_mask: 建筑物掩码，与img相同大小。1表示建筑物位置，0表示非建筑物位置
        """
        B, C, H, W = img.shape
        mask = torch.ones_like(img, device=img.device)  # 初始化为全1（表示所有区域都保留）
        
        # 1. 计算总的需要mask的像素数
        if building_mask is not None:
            building_area = torch.sum(building_mask[0, 0])
            non_building_area = H * W - building_area
            target_pixels = int(non_building_area * self.mask_ratio)
        else:
            target_pixels = int(H * W * self.mask_ratio)
            
        # 2. 计算四分之一大小的方块尺寸
        box_h = H // 2
        box_w = W // 2
        
        # 3. 随机选择一个角落
        corner = np.random.choice(4)  # 0:左上, 1:右上, 2:左下, 3:右下
        if corner == 0:  # 左上角
            t, l = 0, 0
        elif corner == 1:  # 右上角
            t, l = 0, W - box_w
        elif corner == 2:  # 左下角
            t, l = H - box_h, 0
        else:  # 右下角
            t, l = H - box_h, W - box_w
            
        # 4. 检查是否与建筑物重叠（如果有建筑物掩码）
        corner_valid = True
        building_overlap = 0
        if building_mask is not None:
            region = building_mask[0, 0, t:t+box_h, l:l+box_w]
            building_overlap = region.sum().item()
            
            # 如果有建筑物重叠，记录但仍然使用该角落
            if building_overlap > 0:
                print(f"Warning: Corner box overlaps with {building_overlap} building pixels")
        
        # 5. 放置大方块掩码（设置为0，表示这些位置被遮挡）
        mask[:, :, t:t+box_h, l:l+box_w] = 0
        
        # 6. 计算大方块已mask的有效像素数（不包括与建筑物重叠的部分）
        if building_mask is not None:
            # 计算大方块中非建筑区域的像素数
            box_non_building_pixels = box_h * box_w - building_overlap
            # 仍然需要遮挡的非建筑区域像素数 - 确保是整数
            remaining_pixels = int(target_pixels - box_non_building_pixels)
        else:
            box_pixels = box_h * box_w
            remaining_pixels = int(target_pixels - box_pixels)
        
        # 7. 如果需要mask的像素不足或已超出，直接返回
        if remaining_pixels <= 0:
            print(f"Box mask covers more than target pixels, no random masking needed")
            # 如果是建筑物场景，确保建筑物区域不被mask
            if building_mask is not None:
                # 恢复建筑物区域（确保建筑物区域为1，即不被mask）
                mask = torch.where(building_mask > 0, torch.ones_like(mask), mask)
            return mask
        
        # 8. 创建一个标记已mask区域的掩码
        masked_positions = torch.zeros((H, W), device=img.device)
        masked_positions[t:t+box_h, l:l+box_w] = 1  # 标记大方块区域
        
        if building_mask is not None:
            # 标记建筑物区域为已mask，这样就不会在建筑物上随机mask
            masked_positions = masked_positions + building_mask[0, 0]
            # 确保值不超过1
            masked_positions = torch.clamp(masked_positions, 0, 1)
        
        # 9. 生成随机像素mask
        # 找出所有可以mask的像素位置（未被方块或建筑物占用的位置）
        available_positions = torch.where(masked_positions == 0)
        available_count = len(available_positions[0])
        
        print(f"Available positions for random masking: {available_count}")
        print(f"Remaining pixels to mask: {remaining_pixels}")
        
        if available_count <= remaining_pixels:
            # 如果剩余可用像素不足，则全部mask
            mask[:, :, available_positions[0], available_positions[1]] = 0
            print(f"Warning: Not enough available pixels. Masked all {available_count} available pixels.")
        else:
            # 随机选择像素进行mask
            random_indices = np.random.choice(available_count, size=int(remaining_pixels), replace=False)
            y_indices = available_positions[0][random_indices]
            x_indices = available_positions[1][random_indices]
            mask[:, :, y_indices, x_indices] = 0
        
        # 10. 如果是建筑物场景，确保建筑物区域不被mask
        if building_mask is not None:
            # 恢复建筑物区域（确保建筑物区域为1，即不被mask）
            mask = torch.where(building_mask > 0, torch.ones_like(mask), mask)
        
        # 11. 打印统计信息
        total_pixels = H * W
        masked_pixels = total_pixels - torch.sum(mask[0, 0])  # 只计算一个通道
        
        print(f"Total image pixels: {total_pixels}")
        corner_names = ["左上角", "右上角", "左下角", "右下角"]
        print(f"Box placed at {corner_names[corner]} ({t}:{t+box_h}, {l}:{l+box_w})")
        print(f"Box size: {box_h}x{box_w} = {box_h*box_w} pixels")
        
        if building_mask is not None:
            building_area = torch.sum(building_mask[0, 0])
            non_building_area = H * W - building_area
            print(f"Building pixels: {building_area}")
            print(f"Non-building pixels: {non_building_area}")
            print(f"Target masked pixels: {target_pixels}")
            print(f"Actual masked pixels in non-building areas: {masked_pixels}")
            print(f"Non-building masked ratio: {(masked_pixels / non_building_area).item():.2%}")
        else:
            print(f"Target masked pixels: {target_pixels}")
            print(f"Actual masked pixels: {masked_pixels}")
            print(f"Masked ratio: {(masked_pixels / total_pixels).item():.2%}")
        
        return mask

    def _retrieve_single_box_mask(self, img, building_mask=None):
        """生成单个大方块掩码，方块大小可配置
        掩码值为0表示被遮挡区域，值为1表示保留区域
        Args:
            img: 输入图像 [B, C, H, W]
            building_mask: 建筑物掩码，与img相同大小。1表示建筑物位置，0表示非建筑物位置
        """
        B, C, H, W = img.shape
        mask = torch.ones_like(img, device=img.device)  # 初始化为全1（表示所有区域都保留）
        
        # 使用box_size参数作为方块大小，如果没有设置，则使用默认值
        box_h, box_w = self.box_size
        
        # 确保box尺寸不超过图像尺寸
        box_h = min(box_h, H - 2 * self.margin[0])
        box_w = min(box_w, W - 2 * self.margin[1])
        
        # 计算可能的位置范围
        max_y = H - self.margin[0] - box_h
        max_x = W - self.margin[1] - box_w
        
        if max_y <= self.margin[0] or max_x <= self.margin[1]:
            raise ValueError("Margin too large for box mask")
        
        # 尝试放置方块，避开建筑物区域
        attempts = 0
        max_attempts = 50
        box_placed = False
        
        while not box_placed and attempts < max_attempts:
            # 随机选择位置
            t = np.random.randint(self.margin[0], max_y)
            l = np.random.randint(self.margin[1], max_x)
            
            # 检查是否与建筑物重叠
            if building_mask is not None:
                region = building_mask[0, 0, t:t+box_h, l:l+box_w]
                if region.sum() > 0:  # 如果与建筑物有任何重叠，跳过
                    attempts += 1
                    continue
            
            # 放置方块掩码（设置为0，表示这些位置被遮挡）
            mask[:, :, t:t+box_h, l:l+box_w] = 0
            box_placed = True
            break
        
        if not box_placed:
            print(f"Warning: Could not place box mask after {attempts} attempts")
            # 如果多次尝试后仍无法放置不与建筑物重叠的方块，则强制放置
            t = np.random.randint(self.margin[0], max_y)
            l = np.random.randint(self.margin[1], max_x)
            mask[:, :, t:t+box_h, l:l+box_w] = 0
            if building_mask is not None:
                print("Forced placement may overlap with buildings")
        
        # 计算掩码率
        total_pixels = H * W
        masked_pixels = total_pixels - torch.sum(mask[0, 0])  # 只计算一个通道
        mask_ratio = masked_pixels / total_pixels
        
        print(f"Box placed at position ({t}:{t+box_h}, {l}:{l+box_w})")
        print(f"Box size: {box_h}x{box_w} = {box_h*box_w} pixels")
        print(f"Total image pixels: {total_pixels}")
        print(f"Masked pixels: {masked_pixels}")
        print(f"Actual mask ratio: {mask_ratio.item():.2%}")
        
        return mask

    def __call__(self, img, building_mask=None):
        """Generate mask based on specified type"""
        if self.mask_type == 'sensor_rect':  # 规律采样 + 建筑物已知
            return self._retrieve_sensor_rect(img, building_mask)
        elif self.mask_type == 'sensor_rect_uncond':  # 规律采样 + 建筑物未知
            return self._retrieve_sensor_rect_uncond(img, building_mask)
        elif self.mask_type == 'random_cond':  # 随机采样 + 建筑物已知
            return self._retrieve_random_cond(img, building_mask)
        elif self.mask_type == 'random_uncond':  # 随机采样 + 建筑物未知
            return self._retrieve_random_uncond(img, building_mask)
        elif self.mask_type == 'fixed_rect':  # 固定矩形掩码
            return self._retrieve_fixed_rect(img, building_mask)
        elif self.mask_type == 'rect_sample':  # 新方法：矩形采样点（与fixed_rect相反）
            return self._retrieve_rect_sample(img, building_mask)
        elif self.mask_type == 'box_random':  # 新方法：大区域掩码+随机像素掩码
            return self._retrieve_box_random_mask(img, building_mask)
        elif self.mask_type == 'corner_box_random':  # 新方法：角落大方块+随机像素掩码
            return self._retrieve_corner_box_random_mask(img, building_mask)
        elif self.mask_type == 'single_box':  # 新方法：单个大方块掩码
            return self._retrieve_single_box_mask(img, building_mask)
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")

def unnormalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img / scaling


def normalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img * scaling


def dynamic_thresholding(img, s=0.95):
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)


def get_gaussian_kernel(kernel_size=31, std=0.5):
    n = np.zeros([kernel_size, kernel_size])
    n[kernel_size//2, kernel_size//2] = 1
    k = scipy.ndimage.gaussian_filter(n, sigma=std)
    k = k.astype(np.float32)
    return k


def init_kernel_torch(kernel, device="cuda:0"):
    h, w = kernel.shape
    kernel = Variable(torch.from_numpy(kernel).to(device), requires_grad=True)
    kernel = kernel.view(1, 1, h, w)
    kernel = kernel.repeat(1, 3, 1, 1)
    return kernel


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


class exact_posterior():
    def __init__(self, betas, sigma_0, label_dim, input_dim):
        self.betas = betas
        self.sigma_0 = sigma_0
        self.label_dim = label_dim
        self.input_dim = input_dim

    def py_given_x0(self, x0, y, A, verbose=False):
        norm_const = 1/((2 * np.pi)**self.input_dim * self.sigma_0**2)
        exp_in = -1/(2 * self.sigma_0**2) * torch.linalg.norm(y - A(x0))**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def pxt_given_x0(self, x0, xt, t, verbose=False):
        beta_t = self.betas[t]
        norm_const = 1/((2 * np.pi)**self.label_dim * beta_t)
        exp_in = -1/(2 * beta_t) * torch.linalg.norm(xt - np.sqrt(1 - beta_t)*x0)**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def prod_logsumexp(self, x0, xt, y, A, t):
        py_given_x0_density, pyx0_nc, pyx0_ei = self.py_given_x0(x0, y, A, verbose=True)
        pxt_given_x0_density, pxtx0_nc, pxtx0_ei = self.pxt_given_x0(x0, xt, t, verbose=True)
        summand = (pyx0_nc * pxtx0_nc) * torch.exp(-pxtx0_ei - pxtx0_ei)
        return torch.logsumexp(summand, dim=0)



def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def total_variation_loss(img, weight):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).mean()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).mean()
    return weight * (tv_h + tv_w)


if __name__ == '__main__':
    import numpy as np
    from torch import nn
    import matplotlib.pyplot as plt
    device = 'cuda:0'
    load_path = '/media/harry/tomo/FFHQ/256/test/00000.png'
    img = torch.tensor(plt.imread(load_path)[:, :, :3])  #rgb
    img = torch.permute(img, (2, 0, 1)).view(1, 3, 256, 256).to(device)

    mask_len_range = (32, 128)
    mask_prob_range = (0.3, 0.7)
    image_size = 256
    # mask
    mask_gen = mask_generator(
        mask_len_range=mask_len_range,
        mask_prob_range=mask_prob_range,
        image_size=image_size
    )
    mask = mask_gen(img)

    mask = np.transpose(mask.squeeze().cpu().detach().numpy(), (1, 2, 0))

    plt.imshow(mask)
    plt.show()
