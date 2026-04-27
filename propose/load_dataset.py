import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class RandomTransform:
    def __init__(self, cfg):
        self.cfg = cfg

        self.prob = cfg.augmentation.prob

        self.transform_list = []

        self.transform_list.append(T.Resize(size=self.cfg.dataset.image_size))
        
        if self.cfg.augmentation.random_flip:
            self.transform_list.append(T.RandomHorizontalFlip(p=self.prob))
        
        if self.cfg.augmentation.random_affine:
            self.transform_list.append(T.RandomAffine(
                degrees=10,
                shear=(-10, 10, -10, 10)
            ))
        
        if self.cfg.augmentation.color_jitter:
            self.transform_list.append(T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
            ))

        self.transform_list.append(T.ToTensor())
        self.transform_list.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
        
        if self.cfg.augmentation.random_erasing:
            self.transform_list.append(T.RandomErasing(p=self.prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.4))

        self.transform = T.Compose(self.transform_list)
            
    def __call__(self, sample):
        img, v_id, c_id, img_name = sample['img'], sample['v_id'], sample['c_id'], sample['img_name']
        img = self.transform(img)         
        return {
            'img': img,
            'v_id': v_id,
            'c_id': c_id,
            'img_name': img_name
        }

class ValTransform:
    def __init__(self, cfg):
        self.cfg = cfg

        self.transform = T.Compose([
            T.Resize(size=self.cfg.dataset.image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, sample):
        img, v_id, c_id, img_name = sample['img'], sample['v_id'], sample['c_id'], sample['img_name']
        img = self.transform(img)
        return {
            'img': img,
            'v_id': v_id,
            'c_id': c_id,
            'img_name': img_name
        }

class ReIDBaseDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, v_id, c_id, img_path = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Không thể mở ảnh {img_path}: {e}")
            return None
        
        sample = {"img": img, "v_id": v_id, "c_id": c_id, "img_name": img_name}

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

class VeRiDataset(ReIDBaseDataset):
    def __init__(self, data_path: str, transform=None):
        super().__init__(transform)
        self.data_path = data_path
        self.img_list = os.listdir(self.data_path)
        
        for img_name in self.img_list:
            if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                continue
            parts = img_name.split("_")
            v_id = parts[0]
            c_id = parts[1]
            img_path = os.path.join(self.data_path, img_name)
            self.samples.append((img_name, v_id, c_id, img_path))

class VRICDataset(ReIDBaseDataset):
    def __init__(self, data_path: str, list_path: str, transform=None):
        super().__init__(transform)
        self.data_path = data_path
        self.list_path = list_path

        with open(self.list_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                img_name = parts[0]
                v_id = parts[1]
                c_id = parts[2]
                img_path = os.path.join(self.data_path, img_name)
                self.samples.append((img_name, v_id, c_id, img_path))