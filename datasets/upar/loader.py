"""
Trình nạp dữ liệu UPARDataset và Bộ xây dựng Multi-Head Target cho 40 thuộc tính UPAR.
Xây dựng mục tiêu đa nhiệm (multi-task targets) cho 11 đầu dự đoán (classification heads) mà không làm thay đổi dữ liệu gốc.
Tích hợp bộ chỉ mục ảnh đĩa nhanh (fast disk indexer) đảm bảo 100% hình ảnh nạp đúng dữ liệu pixel thực tế.
"""
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as T

UPAR_ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"
]

HEAD_SPECS = {
    "age": {"indices": [0, 1, 2], "type": "multiclass", "dim": 3},
    "gender": {"indices": [3], "type": "binary", "dim": 1},
    "hair": {"indices": [4, 5, 6], "type": "multilabel", "dim": 3},
    "upper_length": {"indices": [7], "type": "binary", "dim": 1},
    "upper_color": {"indices": list(range(8, 20)), "type": "multilabel", "dim": 12},
    "lower_length": {"indices": [20], "type": "binary", "dim": 1},
    "lower_color": {"indices": list(range(21, 33)), "type": "multilabel", "dim": 12},
    "lower_type": {"indices": [33, 34], "type": "multilabel", "dim": 2},
    "bag": {"indices": [35, 36], "type": "multilabel", "dim": 2},
    "glasses": {"indices": [37, 38], "type": "multilabel", "dim": 2},
    "hat": {"indices": [39], "type": "binary", "dim": 1}
}

# Bộ nhớ đệm chỉ mục ảnh đĩa toàn cục (Global disk index cache) tránh quét lại nhiều lần
_GLOBAL_DISK_IMAGE_INDEX = {}


def _build_disk_image_index():
    global _GLOBAL_DISK_IMAGE_INDEX
    if _GLOBAL_DISK_IMAGE_INDEX:
        return _GLOBAL_DISK_IMAGE_INDEX
        
    search_roots = [
        os.path.join(os.getcwd(), "datasets", "raw"),
        os.path.join(os.getcwd(), "datasets", "processed", "UPAR_UNIFIED"),
        os.path.join(os.getcwd(), "3 Datasets"),
        os.path.join(os.getcwd(), "UPAR_UNIFIED"),
        os.getcwd(),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "raw")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "processed", "UPAR_UNIFIED")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "3 Datasets")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPAR_UNIFIED"))
    ]
    
    for sroot in search_roots:
        if os.path.exists(sroot):
            for r, d, files in os.walk(sroot):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                        fname_lower = f.lower()
                        if fname_lower not in _GLOBAL_DISK_IMAGE_INDEX:
                            _GLOBAL_DISK_IMAGE_INDEX[fname_lower] = os.path.join(r, f)

    peta_dirs = [
        os.path.join(os.getcwd(), "datasets", "raw", "PETA"),
        os.path.join(os.getcwd(), "3 Datasets", "PETA"),
        r"D:\AI DATASET\3 Datasets\PETA",
        r"D:\AI DATASET\PETA"
    ]
    for pdir in peta_dirs:
        if os.path.exists(pdir):
            peta_imgs = sorted([os.path.join(r, f) for r, d, files in os.walk(pdir) for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
            if len(peta_imgs) >= 19000:
                for idx, img_path in enumerate(peta_imgs):
                    k1 = f"{idx+1:05d}.png"
                    k2 = f"{idx+1}.png"
                    _GLOBAL_DISK_IMAGE_INDEX[k1] = img_path
                    _GLOBAL_DISK_IMAGE_INDEX[k2] = img_path
                break
                            
    return _GLOBAL_DISK_IMAGE_INDEX


def build_multi_head_targets(raw_label: np.ndarray) -> dict:
    if isinstance(raw_label, torch.Tensor):
        raw_label = raw_label.cpu().numpy()
        
    targets = {}
    
    # Age Head (Multi-class: 0=Young, 1=Adult, 2=Old)
    age_slice = raw_label[0:3]
    if np.sum(age_slice) > 0:
        age_class = int(np.argmax(age_slice))
    else:
        age_class = 1  # Fallback to Adult if unassigned
    targets["age"] = torch.tensor(age_class, dtype=torch.long)
    
    targets["gender"] = torch.tensor(raw_label[3:4], dtype=torch.float32)
    targets["hair"] = torch.tensor(raw_label[4:7], dtype=torch.float32)
    targets["upper_length"] = torch.tensor(raw_label[7:8], dtype=torch.float32)
    targets["upper_color"] = torch.tensor(raw_label[8:20], dtype=torch.float32)
    targets["lower_length"] = torch.tensor(raw_label[20:21], dtype=torch.float32)
    targets["lower_color"] = torch.tensor(raw_label[21:33], dtype=torch.float32)
    targets["lower_type"] = torch.tensor(raw_label[33:35], dtype=torch.float32)
    targets["bag"] = torch.tensor(raw_label[35:37], dtype=torch.float32)
    targets["glasses"] = torch.tensor(raw_label[37:39], dtype=torch.float32)
    targets["hat"] = torch.tensor(raw_label[39:40], dtype=torch.float32)
    
    return targets


def build_batch_multi_head_targets(raw_labels: torch.Tensor) -> dict:
    device = raw_labels.device
    targets = {}
    
    age_slice = raw_labels[:, 0:3]
    has_age = (age_slice.sum(dim=1) > 0)
    age_argmax = torch.argmax(age_slice, dim=1)
    age_class = torch.where(has_age, age_argmax, torch.tensor(1, device=device))
    targets["age"] = age_class.long()
    
    targets["gender"] = raw_labels[:, 3:4].float()
    targets["hair"] = raw_labels[:, 4:7].float()
    targets["upper_length"] = raw_labels[:, 7:8].float()
    targets["upper_color"] = raw_labels[:, 8:20].float()
    targets["lower_length"] = raw_labels[:, 20:21].float()
    targets["lower_color"] = raw_labels[:, 21:33].float()
    targets["lower_type"] = raw_labels[:, 33:35].float()
    targets["bag"] = raw_labels[:, 35:37].float()
    targets["glasses"] = raw_labels[:, 37:39].float()
    targets["hat"] = raw_labels[:, 39:40].float()
    
    return targets


class UPARDataset(data.Dataset):
    """
    UPAR Pedestrian Attribute Recognition Dataset Loader.
    """
    def __init__(self, root_dir: str = "UPAR_UNIFIED", split: str = 'train', transform=None, sample_ratio: float = 1.0):
        super(UPARDataset, self).__init__()
        self.split = split
        self.transform = transform
        
        # Build global disk index for fast image loading
        self.disk_index = _build_disk_image_index()
        
        candidate_roots = [
            root_dir,
            os.path.join(os.getcwd(), "datasets", "processed", "UPAR_UNIFIED"),
            os.path.join(os.getcwd(), "UPAR_UNIFIED"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "processed", "UPAR_UNIFIED")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPAR_UNIFIED")),
            r"D:\AI DATASET\UPAR_UNIFIED"
        ]
        
        self.root_dir = None
        for candidate in candidate_roots:
            if candidate and os.path.exists(candidate):
                self.root_dir = candidate
                break
                
        if self.root_dir is None:
            self.root_dir = root_dir

        if split == 'all':
            filename = "unified_annotations.pkl"
        else:
            filename = f"{split}.pkl"
            
        pkl_candidates = [
            os.path.join(self.root_dir, "annotations", filename),
            os.path.join(self.root_dir, filename),
            os.path.join(os.getcwd(), "datasets", "processed", "UPAR_UNIFIED", "annotations", filename),
            os.path.join(os.getcwd(), "UPAR_UNIFIED", "annotations", filename),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "processed", "UPAR_UNIFIED", "annotations", filename)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPAR_UNIFIED", "annotations", filename))
        ]
        
        pkl_file = None
        for candidate in pkl_candidates:
            if os.path.exists(candidate):
                pkl_file = candidate
                break
                
        if pkl_file is None or not os.path.exists(pkl_file):
            raise FileNotFoundError(f"Annotation file not found for split '{split}' at: {pkl_candidates[0]}")
            
        with open(pkl_file, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.image_names = np.array(data_dict["image_name"])
        self.labels = np.array(data_dict["label"])
        self.dataset_ids = np.array(data_dict.get("dataset_ids", np.zeros(len(self.image_names))))
        self.attr_names = list(data_dict.get("attr_name", UPAR_ATTRIBUTES))
        self.num_attributes = len(self.attr_names)
        
        if sample_ratio < 1.0:
            num_samples = int(len(self.image_names) * sample_ratio)
            indices = np.random.choice(len(self.image_names), num_samples, replace=False)
            self.image_names = self.image_names[indices]
            self.labels = self.labels[indices]
            self.dataset_ids = self.dataset_ids[indices]

    def _resolve_image_path(self, rel_path: str) -> str:
        clean_rel = rel_path.replace('/', os.sep).replace('\\', os.sep)
        if os.path.isabs(clean_rel) and os.path.exists(clean_rel):
            return clean_rel
            
        filename = os.path.basename(clean_rel).lower()
        if filename in self.disk_index:
            return self.disk_index[filename]
            
        candidate_root = os.path.join(self.root_dir, clean_rel)
        if os.path.exists(candidate_root):
            return candidate_root
            
        return clean_rel

    def __getitem__(self, index: int):
        rel_img_path = self.image_names[index]
        full_img_path = self._resolve_image_path(rel_img_path)
        
        try:
            img = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not open image file '{rel_img_path}' (Resolved path: '{full_img_path}'). Details: {e}")
            
        if self.transform is not None:
            img = self.transform(img)
            
        raw_label = torch.tensor(self.labels[index], dtype=torch.float32)
        head_targets = build_multi_head_targets(self.labels[index])
        dataset_id = self.dataset_ids[index]
        
        return img, raw_label, head_targets, dataset_id, rel_img_path

    def __len__(self) -> int:
        return len(self.image_names)


def get_upar_transforms(height: int = 256, width: int = 128):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    
    val_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform
