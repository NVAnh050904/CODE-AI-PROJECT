import sys
import os
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms

def main():
    print("=" * 60)
    print("KIỂM THỬ: KIỂM TRA TRÌNH NẠP DỮ LIỆU MULTI-HEAD UPAR DATASET LOADER")
    print("=" * 60)
    
    train_transform, val_transform = get_upar_transforms(height=256, width=128)
    
    dataset = UPARDataset(split='all', transform=val_transform)
    ds_len = len(dataset)
    print(f"Tổng số mẫu trong Dataset: {ds_len}")
    assert ds_len == 145656, f"Kỳ vọng kích thước dataset 145656, nhận được {ds_len}"
    
    img, raw_label, head_targets, ds_id, img_name = dataset[0]
    print(f"Tên ảnh mẫu: {img_name}")
    print(f"Dataset ID mẫu: {ds_id}")
    print(f"Kích thước ảnh đơn (Single image shape): {img.shape}")
    print(f"Kích thước nhãn thô đơn (Single raw label shape): {raw_label.shape}")
    print(f"Danh sách khóa Head targets: {list(head_targets.keys())}")
    
    assert img.shape == torch.Size([3, 256, 128]), f"Kỳ vọng kích thước ảnh [3, 256, 128], nhận được {img.shape}"
    assert raw_label.shape == torch.Size([40]), f"Kỳ vọng kích thước nhãn [40], nhận được {raw_label.shape}"
    assert len(head_targets) == 11, f"Kỳ vọng 11 đầu dự đoán (head targets), nhận được {len(head_targets)}"
    assert head_targets["age"].dtype == torch.long, "Nhãn Age target phải có kiểu torch.long cho CrossEntropy Loss"
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    batch_imgs, batch_labels, batch_head_targets, batch_ids, batch_names = next(iter(loader))
    
    print(f"Kích thước Batch ảnh: {batch_imgs.shape}")
    print(f"Kích thước Batch nhãn: {batch_labels.shape}")
    
    assert batch_imgs.shape == torch.Size([16, 3, 256, 128]), f"Kỳ vọng kích thước batch ảnh [16, 3, 256, 128], nhận được {batch_imgs.shape}"
    assert batch_labels.shape == torch.Size([16, 40]), f"Kỳ vọng kích thước batch nhãn [16, 40], nhận được {batch_labels.shape}"
    assert len(batch_head_targets) == 11, f"Kỳ vọng 11 batch head targets, nhận được {len(batch_head_targets)}"
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM THỬ DATASET: PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
