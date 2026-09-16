"""
Script suy luận (Inference) trên một ảnh đơn cho mô hình Multi-Head PAR Model (UPAR 40 attributes).
Hiển thị báo cáo cấu trúc theo 11 Heads (Age, Gender, Hair, Upper/Lower Color/Length, Accessories)
và xuất báo cáo trực quan chất lượng cao bằng PIL/Matplotlib.

Cách sử dụng:
    python inference/predict_image.py --image "0007_002.jpg"
    python inference/predict_image.py --image "i-LID/0007_002.jpg"
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

HAS_MATPLOTLIB = False
MATPLOTLIB_ERR = ""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception as e:
    HAS_MATPLOTLIB = False
    MATPLOTLIB_ERR = str(e)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hydraplus.par_model import UnifiedPARModel
from datasets.upar.loader import UPAR_ATTRIBUTES

COLOR_NAMES = [
    "Black", "Blue", "Brown", "Green", "Grey",
    "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"
]


def resolve_image_path(query_path: str) -> str:
    """Smart path resolution with auto-extension & disk index support."""
    query_clean = query_path.strip('"').strip("'").replace('/', os.sep).replace('\\', os.sep)
    
    if os.path.exists(query_clean):
        return os.path.abspath(query_clean)
        
    for ext in ['', '.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
        cand = query_clean + ext
        if os.path.exists(cand):
            return os.path.abspath(cand)

    try:
        from datasets.upar.loader import _build_disk_image_index
        idx_map = _build_disk_image_index()
        fname = os.path.basename(query_clean).lower()
        if fname in idx_map:
            return idx_map[fname]
        for ext in ['.jpg', '.png', '.jpeg']:
            if (fname + ext) in idx_map:
                return idx_map[fname + ext]
    except Exception:
        pass
        
    search_roots = [
        os.path.join(os.getcwd(), "3 Datasets"),
        os.path.join(os.getcwd(), "UPAR_UNIFIED"),
        os.getcwd()
    ]
    
    matches = []
    filename = os.path.basename(query_clean).lower()
    
    for root in search_roots:
        if os.path.exists(root):
            for r, d, files in os.walk(root):
                for f in files:
                    f_lower = f.lower()
                    if f_lower == filename or f_lower.startswith(filename + '.'):
                        full_p = os.path.join(r, f)
                        if full_p not in matches:
                            matches.append(full_p)
                        
    if not matches:
        return None
        
    return matches[0]


def get_default_font(size=14):
    """Load clean TrueType font or fallback."""
    font_candidates = ["arial.ttf", "calibri.ttf", "dejavusans.ttf", "segoeui.ttf"]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def parse_head_predictions(head_logits_dict: dict, threshold: float = 0.5) -> dict:
    """
    Parse 11 head logits into human-readable structured predictions with confidences.
    """
    results = {}
    
    # 1. Age Head (Softmax 3 classes)
    age_probs = torch.softmax(head_logits_dict["age"], dim=1).squeeze(0).cpu().numpy()
    age_idx = np.argmax(age_probs)
    age_labels = ["Young", "Adult", "Old"]
    results["age"] = {"prediction": age_labels[age_idx], "confidence": float(age_probs[age_idx])}
    
    # 2. Gender Head (Sigmoid 1 class)
    gender_prob = torch.sigmoid(head_logits_dict["gender"]).item()
    if gender_prob >= threshold:
        results["gender"] = {"prediction": "Female", "confidence": gender_prob}
    else:
        results["gender"] = {"prediction": "Male", "confidence": 1.0 - gender_prob}
        
    # 3. Hair Head (Sigmoid 3 classes: Short, Long, Bald)
    hair_probs = torch.sigmoid(head_logits_dict["hair"]).squeeze(0).cpu().numpy()
    hair_labels = ["Short", "Long", "Bald"]
    detected_hair = [hair_labels[i] for i, p in enumerate(hair_probs) if p >= threshold]
    if not detected_hair:
        best_i = np.argmax(hair_probs)
        detected_hair = [hair_labels[best_i]]
        conf = float(hair_probs[best_i])
    else:
        conf = float(np.max(hair_probs))
    results["hair"] = {"prediction": ", ".join(detected_hair), "confidence": conf}
    
    # 4. Upper Length Head
    up_len_prob = torch.sigmoid(head_logits_dict["upper_length"]).item()
    if up_len_prob >= threshold:
        results["upper_length"] = {"prediction": "Short Sleeve/Upper", "confidence": up_len_prob}
    else:
        results["upper_length"] = {"prediction": "Long Sleeve/Upper", "confidence": 1.0 - up_len_prob}
        
    # 5. Upper Color Head (12 classes)
    up_col_probs = torch.sigmoid(head_logits_dict["upper_color"]).squeeze(0).cpu().numpy()
    detected_up_cols = [COLOR_NAMES[i] for i, p in enumerate(up_col_probs) if p >= threshold]
    if not detected_up_cols:
        best_i = np.argmax(up_col_probs)
        detected_up_cols = [COLOR_NAMES[best_i]]
        conf = float(up_col_probs[best_i])
    else:
        conf = float(np.max(up_col_probs))
    results["upper_color"] = {"prediction": ", ".join(detected_up_cols), "confidence": conf}
    
    # 6. Lower Length Head
    low_len_prob = torch.sigmoid(head_logits_dict["lower_length"]).item()
    if low_len_prob >= threshold:
        results["lower_length"] = {"prediction": "Short Lower", "confidence": low_len_prob}
    else:
        results["lower_length"] = {"prediction": "Long Lower", "confidence": 1.0 - low_len_prob}
        
    # 7. Lower Color Head (12 classes)
    low_col_probs = torch.sigmoid(head_logits_dict["lower_color"]).squeeze(0).cpu().numpy()
    detected_low_cols = [COLOR_NAMES[i] for i, p in enumerate(low_col_probs) if p >= threshold]
    if not detected_low_cols:
        best_i = np.argmax(low_col_probs)
        detected_low_cols = [COLOR_NAMES[best_i]]
        conf = float(low_col_probs[best_i])
    else:
        conf = float(np.max(low_col_probs))
    results["lower_color"] = {"prediction": ", ".join(detected_low_cols), "confidence": conf}
    
    # 8. Lower Type Head (2 classes: Trousers&Shorts, Skirt&Dress)
    low_type_probs = torch.sigmoid(head_logits_dict["lower_type"]).squeeze(0).cpu().numpy()
    type_labels = ["Trousers/Shorts", "Skirt/Dress"]
    detected_types = [type_labels[i] for i, p in enumerate(low_type_probs) if p >= threshold]
    if not detected_types:
        best_i = np.argmax(low_type_probs)
        detected_types = [type_labels[best_i]]
        conf = float(low_type_probs[best_i])
    else:
        conf = float(np.max(low_type_probs))
    results["lower_type"] = {"prediction": ", ".join(detected_types), "confidence": conf}
    
    # 9. Bag Head (2 classes: Backpack, Bag)
    bag_probs = torch.sigmoid(head_logits_dict["bag"]).squeeze(0).cpu().numpy()
    bag_labels = ["Backpack", "Bag/Handbag"]
    detected_bags = [bag_labels[i] for i, p in enumerate(bag_probs) if p >= threshold]
    results["bag"] = {"prediction": ", ".join(detected_bags) if detected_bags else "No Bag", "confidence": float(np.max(bag_probs))}
    
    # 10. Glasses Head (2 classes: Normal, Sun)
    glasses_probs = torch.sigmoid(head_logits_dict["glasses"]).squeeze(0).cpu().numpy()
    glasses_labels = ["Normal Glasses", "Sunglasses"]
    detected_glasses = [glasses_labels[i] for i, p in enumerate(glasses_probs) if p >= threshold]
    results["glasses"] = {"prediction": ", ".join(detected_glasses) if detected_glasses else "No Glasses", "confidence": float(np.max(glasses_probs))}
    
    # 11. Hat Head
    hat_prob = torch.sigmoid(head_logits_dict["hat"]).item()
    if hat_prob >= threshold:
        results["hat"] = {"prediction": "Wearing Hat", "confidence": hat_prob}
    else:
        results["hat"] = {"prediction": "No Hat", "confidence": 1.0 - hat_prob}
        
    return results


def visualize_with_pil(raw_img: Image.Image, img_name: str, probs: np.ndarray, attr_names: list, threshold: float = 0.5, save_path: str = "result_prediction.png"):
    """
    High-quality PIL Bar Chart Renderer.
    """
    sorted_indices = np.argsort(probs)[::-1]
    top_indices = [i for i in sorted_indices if probs[i] >= threshold]
    if len(top_indices) < 8:
        top_indices = list(sorted_indices[:10])
        
    plot_items = [(attr_names[i], probs[i]) for i in top_indices]

    img_w, img_h = raw_img.size
    target_img_h = max(480, len(plot_items) * 45 + 100)
    aspect_ratio = img_w / img_h
    target_img_w = int(target_img_h * aspect_ratio)

    resized_img = raw_img.resize((target_img_w, target_img_h), Image.Resampling.LANCZOS)

    chart_w = 680
    canvas_w = target_img_w + chart_w + 50
    canvas_h = target_img_h + 40

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(resized_img, (20, 20))

    draw = ImageDraw.Draw(canvas)
    
    font_title = get_default_font(18)
    font_sub = get_default_font(13)
    font_label = get_default_font(13)
    font_val = get_default_font(12)

    x_chart = target_img_w + 40
    y_chart = 30
    draw.text((x_chart, y_chart), f"Multi-Head PAR Prediction: {img_name}", fill=(30, 30, 30), font=font_title)
    y_chart += 30
    draw.text((x_chart, y_chart), f"Predicted Attributes (Threshold = {int(threshold*100)}%)", fill=(100, 100, 100), font=font_sub)
    y_chart += 40

    bar_max_w = 320
    bar_start_x = x_chart + 240
    thresh_x = bar_start_x + int(bar_max_w * threshold)

    for name, prob in plot_items:
        is_pos = prob >= threshold
        color = (46, 204, 113) if is_pos else (231, 76, 60)

        draw.text((x_chart, y_chart + 2), name, fill=(40, 40, 40), font=font_label)
        draw.rectangle([bar_start_x, y_chart, bar_start_x + bar_max_w, y_chart + 20], fill=(238, 238, 238))

        fill_w = int(bar_max_w * prob)
        if fill_w > 0:
            draw.rectangle([bar_start_x, y_chart, bar_start_x + fill_w, y_chart + 20], fill=color)

        percent_str = f"{prob * 100:.1f}%"
        draw.text((bar_start_x + bar_max_w + 12, y_chart + 2), percent_str, fill=(30, 30, 30), font=font_val)
        y_chart += 38

    for y_line in range(70, y_chart, 6):
        draw.line([(thresh_x, y_line), (thresh_x, y_line + 3)], fill=(211, 84, 0), width=2)
    draw.text((thresh_x - 30, y_chart + 10), f"Threshold {int(threshold*100)}%", fill=(211, 84, 0), font=font_val)

    canvas.save(save_path)
    print(f"\n[Visual Report Saved]: {os.path.abspath(save_path)}")


def resolve_checkpoint_path(ckpt_arg: str = None) -> str:
    candidates = []
    if ckpt_arg:
        candidates.append(ckpt_arg)
    candidates += [
        os.path.join(os.getcwd(), "checkpoints", "hydraplus_upar_best.pth"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "hydraplus_upar_best.pth")),
        r"D:\AI DATASET\UPAR_UNIFIED\checkpoints\hydraplus_upar_best.pth",
        r"D:\AI DATASET\weights\checkpoints\hydraplus_upar_best.pth",
        r"D:\AI DATASET\weights\market.pth"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return ckpt_arg


def main():
    img_arg = None
    for idx, arg in enumerate(sys.argv):
        if arg.startswith('--image='):
            img_arg = arg.split('=', 1)[1]
        elif arg in ['--image', '-i'] and idx + 1 < len(sys.argv):
            img_arg = sys.argv[idx + 1]

    parser = argparse.ArgumentParser(description="Predict Pedestrian Attributes using Multi-Head PAR Model")
    parser.add_argument('--image', '-i', type=str, default=img_arg, help="Path or filename of pedestrian image")
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Classification probability threshold (default 0.5)")
    args = parser.parse_args()

    if not args.image:
        print("LOI: Vui long cung cap duong dan anh (--image)!")
        sys.exit(1)

    img_path = resolve_image_path(args.image)
    if img_path is None or not os.path.exists(img_path):
        print(f"LOI: Khong tim thay file anh '{args.image}'!")
        sys.exit(1)

    ckpt_path = resolve_checkpoint_path(args.checkpoint)

    print("=" * 70)
    print("MULTI-HEAD PAR MODEL SINGLE IMAGE INFERENCE")
    print("=" * 70)
    print(f"File anh   : {img_path}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Threshold  : {args.threshold}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UnifiedPARModel(num_attributes=40, backbone_name='resnet50', pretrained=True).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: Checkpoint not found at '{ckpt_path}', using initialized weights.")
    model.eval()

    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_dict = model(input_tensor)
        probs = model.predict_40_probabilities(logits_dict).squeeze(0).cpu().numpy()

    head_results = parse_head_predictions(logits_dict, threshold=args.threshold)

    print("\n" + "=" * 70)
    print("MULTI-HEAD ATTRIBUTE RECOGNITION SUMMARY")
    print("=" * 70)
    print(f"| {'Head Name':<18} | {'Predicted Attribute(s)':<30} | {'Confidence':<10} |")
    print("|--------------------|--------------------------------|------------|")
    for head_name, res in head_results.items():
        print(f"| {head_name:<18} | {res['prediction']:<30} | {res['confidence']*100:>8.2f}%  |")

    print("\n" + "=" * 70)
    print("DETAILED 40 ATTRIBUTE PROBABILITIES")
    print("=" * 70)
    for idx, attr_name in enumerate(UPAR_ATTRIBUTES):
        p = probs[idx]
        status = "[CO] YES" if p >= args.threshold else "     NO"
        print(f"  [{idx:02d}] {attr_name:<32} : {p*100:>6.2f}%  -> {status}")

    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    save_filename = f"result_{os.path.basename(img_path)}.png"
    save_filepath = os.path.join(reports_dir, save_filename)
    visualize_with_pil(raw_img, os.path.basename(img_path), probs, UPAR_ATTRIBUTES, threshold=args.threshold, save_path=save_filepath)


if __name__ == '__main__':
    main()
