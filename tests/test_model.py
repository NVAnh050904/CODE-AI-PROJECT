import sys
import os
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms
from models.hydraplus.par_model import UnifiedPARModel

def main():
    print("=" * 60)
    print("KIỂM THỬ: KIỂM TRA FORWARD PASS CỦA MÔ HÌNH MULTI-HEAD")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị (Device): {device}")
    
    train_transform, _ = get_upar_transforms(height=256, width=128)
    dataset = UPARDataset(split='train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    images, raw_labels, head_targets, _, _ = next(iter(loader))
    images = images.to(device)
    raw_labels = raw_labels.to(device)
    
    print(f"Kích thước Batch ảnh đầu vào (Input images shape): {images.shape}")
    print(f"Kích thước Batch nhãn đầu vào (Input labels shape): {raw_labels.shape}")
    
    model = UnifiedPARModel(num_attributes=40, backbone_name='resnet50', pretrained=True).to(device)
    model.eval()
    
    with torch.no_grad():
        outputs_dict = model(images)
        probs_40 = model.predict_40_probabilities(outputs_dict)
        logits_40 = model.predict_40_logits(outputs_dict)
        
    print(f"Danh sách khóa đầu ra mô hình ({len(outputs_dict)} heads): {list(outputs_dict.keys())}")
    for h_name, h_logits in outputs_dict.items():
        print(f"  - Head '{h_name}': kích thước {h_logits.shape}")
        
    print(f"Kích thước xác suất dự đoán 40 thuộc tính (Probabilities shape): {probs_40.shape}")
    print(f"Kích thước Logits dự đoán 40 thuộc tính (Logits shape): {logits_40.shape}")
    
    assert len(outputs_dict) == 11, f"Kỳ vọng 11 heads, nhận được {len(outputs_dict)}"
    assert outputs_dict["age"].shape == torch.Size([8, 3]), f"Kỳ vọng kích thước age head [8, 3], nhận được {outputs_dict['age'].shape}"
    assert outputs_dict["upper_color"].shape == torch.Size([8, 12]), f"Kỳ vọng kích thước upper_color head [8, 12], nhận được {outputs_dict['upper_color'].shape}"
    assert probs_40.shape == torch.Size([8, 40]), f"Kỳ vọng kích thước probs_40 [8, 40], nhận được {probs_40.shape}"
    assert logits_40.shape == torch.Size([8, 40]), f"Kỳ vọng kích thước logits_40 [8, 40], nhận được {logits_40.shape}"
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM THỬ MODEL: PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
