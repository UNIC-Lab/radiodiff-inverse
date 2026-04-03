from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os
from torchvision import transforms as T
from pathlib import Path


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img


@register_dataset(name='building')
class BuildingDataset(VisionDataset):
    def __init__(self, root: str, image_dir: str = None, building_dir: str = None, transforms: Optional[Callable]=None, num_buildings=50, num_sources=4):
        """
        Args:
            root: 数据根目录
            image_dir: DPM图像目录
            building_dir: 建筑物掩码目录
            transforms: 图像变换
            num_buildings: 使用的建筑物场景数量（默认50）
            num_sources: 每个建筑物场景使用的信号源数量（默认4）
        """
        # 在调用父类初始化之前创建空间变换
        self.spatial_transforms = T.Compose([
            T.ToTensor(),
            T.CenterCrop((256, 256)),
            T.Resize((256, 256))
        ])
        
        super().__init__(root, transforms)
        
        self.image_dir = Path(root) / (image_dir if image_dir else 'images')
        self.building_dir = Path(root) / (building_dir if building_dir else 'buildings')
        
        # 获取所有建筑物ID
        building_files = sorted(list(self.building_dir.glob('*.png')))[:num_buildings]
        self.building_ids = [f.stem for f in building_files]
        
        # 构建图像路径列表
        self.image_paths = []
        for building_id in self.building_ids:
            pattern = f"{building_id}_*.png"
            matching_files = sorted(list(self.image_dir.glob(pattern)))[:num_sources]
            self.image_paths.extend(str(f) for f in matching_files)
        
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")
        print(f"Building masks directory: {self.building_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # 获取DPM图像路径
        img_path = self.image_paths[index]
        building_id = os.path.basename(img_path).split('_')[0]
        building_path = os.path.join(self.building_dir, f"{building_id}.png")
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        building_mask = Image.open(building_path).convert('L')
        
        # 对图像应用完整的转换
        if self.transforms:
            image = self.transforms(image)
        
        # 对建筑物掩码只应用空间转换
        building_mask = self.spatial_transforms(building_mask)
        # 二值化掩码并扩展到3个通道
        building_mask = (building_mask > 0.5).float()
        building_mask = building_mask.repeat(3, 1, 1)
        
        return image, building_mask, img_path