import sys
import os
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms, build_batch_multi_head_targets
from models.hydraplus.par_model import UnifiedPARModel
from training.loss import MultiHeadPARLoss

def main():
    print("=" * 60)
    print("KIỂM THỬ: KIỂM TRA LUỒNG MULTI-HEAD FORWARD + LOSS COMPUTATION + BACKWARD PASS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị (Device): {device}")
    
    train_transform, _ = get_upar_transforms(height=256, width=128)
    dataset = UPARDataset(split='train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    batch = next(iter(loader))
    images = batch[0].to(device)
    raw_labels = batch[1].to(device)
    head_targets = build_batch_multi_head_targets(raw_labels)
    
    # 1. Lan truyền tiến (Forward Pass)
    model = UnifiedPARModel(num_attributes=40, backbone_name='resnet50', pretrained=True).to(device)
    model.train()
    
    outputs_dict = model(images)
    print(f"Lan truyền tiến (Forward Pass): PASS (Danh sách đầu ra của {len(outputs_dict)} heads)")
    assert len(outputs_dict) == 11, "Kỳ vọng 11 đầu dự đoán (head outputs)!"
    
    # 2. Tính toán Hàm mất mát (Loss Computation)
    criterion = MultiHeadPARLoss().to(device)
    total_loss, head_losses = criterion.compute_losses(outputs_dict, head_targets)
    print(f"Tính toán Loss Pass: PASS (Tổng loss: {total_loss.item():.4f})")
    for h_name, h_val in head_losses.items():
        print(f"  - Head '{h_name:<15}' loss: {h_val.item():.4f}")
        assert not torch.isnan(h_val) and not torch.isinf(h_val), f"Loss của head '{h_name}' bị NaN hoặc Inf!"
        
    assert not torch.isnan(total_loss) and not torch.isinf(total_loss), "Tổng loss bị NaN hoặc Inf!"
    
    # 3. Lan truyền ngược (Backward Pass)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    
    grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_count += 1
    print(f"Lan truyền ngược (Backward Pass): PASS ({grad_count} tham số nhận được gradients)")
    assert grad_count > 0, "Không có gradient nào được tính toán!"
    
    optimizer.step()
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM THỬ HUẤN LUYỆN: PASS (Forward: PASS, Loss: PASS, Backward: PASS)")
    print("=" * 60)

if __name__ == "__main__":
    main()
