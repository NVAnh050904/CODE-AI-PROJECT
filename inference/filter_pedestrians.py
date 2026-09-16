"""
Hệ thống Lọc và Tìm kiếm Người đi bộ theo Thuộc tính (Attribute-Based Pedestrian Search & Filtering)
Cho phép truy vấn và lọc danh sách đối tượng dựa trên tổ hợp các đặc điểm (Giới tính, Áo, Quần, Tóc, Phụ kiện, Độ tuổi...).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_gallery_database(csv_path=None):
    if csv_path is None or not os.path.exists(csv_path):
        candidates = [
            os.path.join(os.getcwd(), "reports", "csv_full", "test.csv"),
            os.path.join(os.getcwd(), "reports", "csv_full", "val.csv"),
            os.path.join(os.getcwd(), "UPAR_UNIFIED", "reports", "csv_50pct", "test_50pct.csv"),
            os.path.join(os.getcwd(), "reports", "test_50pct.csv"),
            r"c:\Users\ADMIN\OneDrive\Documents\GitHub\AI-Project\UPAR_UNIFIED\reports\csv_50pct\test_50pct.csv"
        ]

        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break

    if csv_path is None or not os.path.exists(csv_path):
        raise FileNotFoundError("Khong tim thay file du lieu CSV (test_50pct.csv) de tim kiem!")

    print(f"[Database]: Dang nap thu vien du lieu tu '{csv_path}'...")
    df = pd.read_csv(csv_path)
    return df, csv_path


def resolve_img_full_path(rel_path):
    search_roots = [
        os.path.join(os.getcwd(), "3 Datasets"),
        os.path.join(os.getcwd(), "UPAR_UNIFIED"),
        os.getcwd()
    ]
    filename = os.path.basename(rel_path)
    for root in search_roots:
        if os.path.exists(root):
            for r, d, files in os.walk(root):
                if filename in files:
                    return os.path.join(r, filename)
    return None


def filter_pedestrians(df, criteria: dict):
    """
    criteria format: {'Gender-Female': 1, 'UpperBody-Color-Red': 1, ...}
    """
    filtered_df = df.copy()
    for attr, val in criteria.items():
        if attr in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[attr] == val]

    return filtered_df


def visualize_filter_gallery(results_df, criteria_str, max_display=10, save_path=None):
    if results_df.empty:
        print("[Visual Gallery]: Khong co anh nao khop de ve gallery.")
        return

    n_samples = min(len(results_df), max_display)
    thumb_w, thumb_h = 128, 256
    padding = 15
    header_h = 70
    grid_cols = min(5, n_samples)
    grid_rows = (n_samples + grid_cols - 1) // grid_cols

    canvas_w = grid_cols * (thumb_w + padding) + padding
    canvas_h = header_h + grid_rows * (thumb_h + padding + 25) + padding

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_sub = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font_title = font_sub = ImageFont.load_default()

    draw.rectangle([(0, 0), (canvas_w, header_h - 10)], fill=(30, 41, 59))
    draw.text((15, 12), f"PEDESTRIAN ATTRIBUTE FILTER RESULTS ({n_samples}/{len(results_df)} MATCHES)", fill=(255, 255, 255), font=font_title)
    draw.text((15, 38), f"Criteria: {criteria_str}", fill=(203, 213, 225), font=font_sub)

    for idx in range(n_samples):
        row = results_df.iloc[idx]
        img_rel = row['image_name']
        img_full = resolve_img_full_path(img_rel)

        r = idx // grid_cols
        c = idx % grid_cols
        x = padding + c * (thumb_w + padding)
        y = header_h + r * (thumb_h + padding + 25)

        if img_full and os.path.exists(img_full):
            try:
                img = Image.open(img_full).convert('RGB')
                img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                canvas.paste(img, (x, y))
                draw.rectangle([(x, y), (x + thumb_w, y + thumb_h)], outline=(59, 130, 246), width=2)
            except Exception:
                draw.rectangle([(x, y), (x + thumb_w, y + thumb_h)], fill=(200, 200, 200))
        else:
            draw.rectangle([(x, y), (x + thumb_w, y + thumb_h)], fill=(220, 220, 220))
            draw.text((x + 10, y + thumb_h // 2), "Image N/A", fill=(100, 100, 100), font=font_sub)

        label_txt = f"#{idx+1}: {os.path.basename(img_rel)[:15]}"
        draw.text((x, y + thumb_h + 4), label_txt, fill=(30, 41, 59), font=font_sub)

    if save_path is None:
        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, "filter_result.png")

    canvas.save(save_path)
    print(f"\n[Visual Gallery Saved]: {os.path.abspath(save_path)}")


def main():
    parser = argparse.ArgumentParser(description="Filter Pedestrians by Attributes")
    parser.add_argument('--gender', type=str, choices=['female', 'male'], help="Filter gender (female / male)")
    parser.add_argument('--age', type=str, choices=['young', 'adult', 'old'], help="Filter age group")
    parser.add_argument('--upper_color', type=str, help="Filter upper body color (Black, Red, Blue, White, etc.)")
    parser.add_argument('--lower_color', type=str, help="Filter lower body color (Black, Blue, Grey, White, etc.)")
    parser.add_argument('--hair', type=str, choices=['short', 'long', 'bald'], help="Filter hair length")
    parser.add_argument('--lower_type', type=str, choices=['trousers', 'skirt'], help="Filter lower body type (trousers / skirt)")
    parser.add_argument('--bag', type=str, choices=['backpack', 'bag', 'nobag'], help="Filter accessory bag")
    parser.add_argument('--max_results', type=int, default=10, help="Max results to display in visual gallery")
    parser.add_argument('--csv_database', type=str, default=None, help="Path to database CSV file")

    args = parser.parse_args()

    df, csv_used = load_gallery_database(args.csv_database)

    criteria = {}
    criteria_desc = []

    if args.gender == 'female':
        criteria['Gender-Female'] = 1
        criteria_desc.append("Gender=Female")
    elif args.gender == 'male':
        criteria['Gender-Female'] = 0
        criteria_desc.append("Gender=Male")

    if args.age == 'young':
        criteria['Age-Young'] = 1
        criteria_desc.append("Age=Young")
    elif args.age == 'adult':
        criteria['Age-Adult'] = 1
        criteria_desc.append("Age=Adult")
    elif args.age == 'old':
        criteria['Age-Old'] = 1
        criteria_desc.append("Age=Old")

    if args.hair == 'short':
        criteria['Hair-Length-Short'] = 1
        criteria_desc.append("Hair=Short")
    elif args.hair == 'long':
        criteria['Hair-Length-Long'] = 1
        criteria_desc.append("Hair=Long")
    elif args.hair == 'bald':
        criteria['Hair-Length-Bald'] = 1
        criteria_desc.append("Hair=Bald")

    if args.upper_color:
        col = f"UpperBody-Color-{args.upper_color.capitalize()}"
        if col in df.columns:
            criteria[col] = 1
            criteria_desc.append(f"UpperColor={args.upper_color.capitalize()}")

    if args.lower_color:
        col = f"LowerBody-Color-{args.lower_color.capitalize()}"
        if col in df.columns:
            criteria[col] = 1
            criteria_desc.append(f"LowerColor={args.lower_color.capitalize()}")

    if args.lower_type == 'skirt':
        criteria['LowerBody-Type-Skirt&Dress'] = 1
        criteria_desc.append("LowerType=Skirt")
    elif args.lower_type == 'trousers':
        criteria['LowerBody-Type-Trousers&Shorts'] = 1
        criteria_desc.append("LowerType=Trousers/Shorts")

    if args.bag == 'backpack':
        criteria['Accessory-Backpack'] = 1
        criteria_desc.append("Accessory=Backpack")
    elif args.bag == 'bag':
        criteria['Accessory-Bag'] = 1
        criteria_desc.append("Accessory=Bag")

    if not criteria:
        print("\n[Huong dan su dung]: Vui long cung cap it nhat 1 dieu kien loc! Vi du:")
        print("  python inference/filter_pedestrians.py --gender female --upper_color red")
        print("  python inference/filter_pedestrians.py --age young --bag backpack")
        sys.exit(0)

    crit_str = " | ".join(criteria_desc)
    filtered = filter_pedestrians(df, criteria)

    print("\n" + "=" * 70)
    print("ATTRIBUTE-BASED PEDESTRIAN SEARCH & FILTERING")
    print("=" * 70)
    print(f"Dieu kien loc  : {crit_str}")
    print(f"Tong so tim thay : {len(filtered):,} / {len(df):,} mau")
    print("=" * 70)

    if not filtered.empty:
        print("\nDANH SACH KER QUA HANG DAU (TOP MATCHES):")
        display_cols = ['image_name', 'dataset_name'] + list(criteria.keys())
        print(filtered[display_cols].head(args.max_results).to_string(index=False))
        visualize_filter_gallery(filtered, crit_str, max_display=args.max_results)
    else:
        print("\nKhông tìm thấy đối tượng nào khớp chính xác với tất cả các điều kiện lọc trên.")


if __name__ == '__main__':
    main()
