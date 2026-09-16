"""
tracking/query_persons.py
==========================
Level 5 - Video Person Retrieval Engine: Công cụ Lọc Thuộc tính UPAR & Truy vấn Ảnh mẫu Re-ID.

Truy vấn CSDL trung tâm reports/tracking/person_database.json hỗ trợ:
1. Lọc theo thuộc tính UPAR (Multi-label heads và Single-label top1).
2. Truy vấn ảnh mẫu Re-ID (Tính Cosine Similarity trên OSNet 512-dim embedding).
3. Kết hợp Attribute Filtering + Re-ID Ranking.
4. Tự động gom nhóm khử trùng lặp cá nhân bằng identity_group_id.
5. Hiển thị bảng kết quả Terminal & Tự động xuất lưới ảnh kết quả (Image Grid).
6. Xử lý các trường hợp biên (Edge Cases: 0 kết quả, thiếu điều kiện tìm kiếm, ảnh query không hợp lệ).
7. Đưa ra các thông báo lưu ý bắt buộc.
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

# Cấu hình UTF-8 stdout chống lỗi charmap encoding trên Windows Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm đường dẫn gốc dự án vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.reid_embedding import ReIDExtractor, DEFAULT_CHECKPOINT_PATH

DATABASE_PATH = PROJECT_ROOT / "reports" / "tracking" / "person_database.json"
QUERY_RESULTS_DIR = PROJECT_ROOT / "reports" / "tracking" / "query_results"


def load_database() -> dict:
    """Load person_database.json. If missing, attempt to build it first."""
    if not DATABASE_PATH.exists():
        print("[NOTICE] person_database.json not found. Running database builder...")
        from tracking.build_person_database import rebuild_all_database
        return rebuild_all_database()

    with open(DATABASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_active_conditions(args) -> dict:
    """Collect active search filters provided by the user."""
    conds = {}
    if args.gender: conds["gender"] = args.gender
    if args.age: conds["age"] = args.age
    if args.hair: conds["hair"] = args.hair
    if args.glasses: conds["glasses"] = args.glasses
    if args.hat: conds["hat"] = args.hat
    if args.upper_color: conds["upper_color"] = args.upper_color
    if args.lower_color: conds["lower_color"] = args.lower_color
    if args.lower_type: conds["lower_type"] = args.lower_type
    if args.bag: conds["bag"] = args.bag
    if args.query_image: conds["query_image"] = args.query_image
    return conds


def match_multi_label(head_list: list, query_val: str) -> bool:
    """
    Match query_val against multi-label head active list (list of dicts with 'label').
    Case-insensitive matching.
    """
    if not query_val:
        return True
    q = query_val.strip().lower()
    for item in head_list:
        label = item.get("label", "").lower()
        if q in label or label in q:
            return True
        # Special mappings
        if q == "backpack" and "bag" in label:
            return True
        if q == "bag" and "bag" in label:
            return True
        if q in ["trousers", "shorts"] and "trousers" in label:
            return True
        if q in ["skirt", "dress"] and "skirt" in label:
            return True
        if q in ["normal", "glasses"] and "glasses" in label:
            return True
        if q in ["sun", "sunglasses"] and "sun" in label:
            return True
    return False


def match_single_label(top1_dict: dict, head_name: str, query_val: str) -> bool:
    """Match query_val against top1_summary head."""
    if not query_val:
        return True
    actual = str(top1_dict.get(head_name, "")).strip().lower()
    q = query_val.strip().lower()
    
    if head_name == "gender":
        return q == actual

    if head_name == "age":
        return q == actual or q in actual

    if head_name == "hat":
        if q in ["hat", "yes", "true"] and actual == "hat":
            return True
        if q in ["no hat", "no_hat", "none", "no", "false"] and actual == "no hat":
            return True
        return q == actual

    return q == actual or q in actual


def filter_by_attributes(database: dict, args) -> list:
    """Filter database records based on CLI attribute parameters."""
    filtered = []
    for rec in database.values():
        top1 = rec.get("top1_summary", {})
        multi = rec.get("multi_label_heads", {})

        # Single-label heads (use top1_summary)
        if args.gender and not match_single_label(top1, "gender", args.gender):
            continue
        if args.age and not match_single_label(top1, "age", args.age):
            continue
        if args.hat and not match_single_label(top1, "hat", args.hat):
            continue

        # Multi-label heads (use multi_label_heads)
        if args.hair and not match_multi_label(multi.get("hair", []), args.hair):
            continue
        if args.glasses and not match_multi_label(multi.get("glasses", []), args.glasses):
            continue
        if args.upper_color and not match_multi_label(multi.get("upper_color", []), args.upper_color):
            continue
        if args.lower_color and not match_multi_label(multi.get("lower_color", []), args.lower_color):
            continue
        if args.lower_type and not match_multi_label(multi.get("lower_type", []), args.lower_type):
            continue
        if args.bag and not match_multi_label(multi.get("bag", []), args.bag):
            continue

        filtered.append(rec)

    return filtered


def rank_by_reid_similarity(records: list, query_image_path: str, extractor: ReIDExtractor) -> list:
    """
    Trích xuất embedding từ ảnh query mẫu và xếp hạng danh sách theo Cosine Similarity.
    Trả về danh sách bản ghi bổ sung trường 'similarity_score'.
    """
    img_p = Path(query_image_path)
    if not img_p.exists():
        print(f"\n[LỖI] Tệp ảnh query-image không tồn tại tại: '{query_image_path}'")
        sys.exit(1)

    print(f"[INFO] Đang trích xuất Re-ID embedding cho ảnh query: '{img_p}'...")
    try:
        query_feats = extractor.extract_features([str(img_p)])  # (1, 512)
    except Exception as e:
        print(f"\n[LỖI] Không thể đọc hoặc giải mã tệp ảnh query-image tại: '{query_image_path}' ({e})")
        sys.exit(1)

    if query_feats is None or len(query_feats) == 0:
        print(f"\n[LỖI] Không thể trích xuất đặc trưng Re-ID cho tệp ảnh: '{query_image_path}'")
        sys.exit(1)

    q_emb = query_feats[0]  # L2 normalized 512 float

    scored_records = []
    for rec in records:
        emb = rec.get("embedding", [])
        if len(emb) == 512:
            t_emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(t_emb)
            if norm > 0:
                t_emb = t_emb / norm
            sim = float(np.dot(q_emb, t_emb))
            sim = float(np.clip(sim, 0.0, 1.0))
        else:
            sim = 0.0

        r_copy = dict(rec)
        r_copy["similarity_score"] = sim
        scored_records.append(r_copy)

    scored_records.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored_records


def deduplicate_by_identity_group(records: list) -> list:
    """
    Group records by identity_group_id.
    Collapse multiple tracks of the same ground-truth identity into 1 single output record,
    annotating track count and selecting highest similarity score / representative crop.
    """
    grouped = defaultdict(list)
    for rec in records:
        group_id = rec.get("identity_group_id", rec.get("global_id"))
        grouped[group_id].append(rec)

    deduped = []
    for group_id, group_recs in grouped.items():
        # Sort by similarity_score if present, else by n_frames or track_id
        group_recs.sort(key=lambda r: (r.get("similarity_score", 0.0), r.get("last_seen_frame", 0)), reverse=True)

        primary = dict(group_recs[0])
        n_tracks = len(group_recs)
        primary["matched_track_count"] = n_tracks
        
        if n_tracks > 1:
            primary["note"] = f"(xuat hien duoi {n_tracks} track do bi che khuat/mat dau)"
        else:
            primary["note"] = ""

        deduped.append(primary)

    # Re-sort deduplicated list
    if any("similarity_score" in r for r in deduped):
        deduped.sort(key=lambda r: r.get("similarity_score", 0.0), reverse=True)

    return deduped


def generate_result_grid_image(records: list, output_filename: str) -> str:
    """Generate result grid image using PIL combining representative crops."""
    from PIL import Image, ImageDraw, ImageFont

    QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QUERY_RESULTS_DIR / output_filename

    if not records:
        return ""

    card_w, card_h = 220, 340
    crop_w, crop_h = 200, 240
    cols = min(5, len(records))
    rows = (len(records) + cols - 1) // cols

    canvas_w = cols * card_w + 40
    canvas_h = rows * card_h + 60

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 24, 30))
    draw = ImageDraw.Draw(canvas)

    # Header title
    draw.text((20, 15), f"PERSON RETRIEVAL RESULTS (Top {len(records)} Identities)", fill=(240, 240, 240))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, rec in enumerate(records):
        col = idx % cols
        row = idx // cols
        x = 20 + col * card_w + 10
        y = 50 + row * card_h + 10

        # Draw card bg
        draw.rectangle([x, y, x + crop_w, y + card_h - 20], fill=(35, 42, 52), outline=(60, 75, 95), width=2)

        # Paste crop
        crop_rel = rec.get("representative_crop", "")
        crop_full = PROJECT_ROOT / crop_rel if crop_rel else None

        if crop_full and crop_full.exists():
            try:
                c_img = Image.open(crop_full).convert("RGB")
                c_img = c_img.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
                canvas.paste(c_img, (x, y))
            except Exception:
                draw.rectangle([x, y, x + crop_w, y + crop_h], fill=(50, 50, 50))
                draw.text((x + 20, y + 100), "Error loading crop", fill=(200, 200, 200), font=font)
        else:
            draw.rectangle([x, y, x + crop_w, y + crop_h], fill=(50, 50, 50))
            draw.text((x + 40, y + 100), "No crop image", fill=(200, 200, 200), font=font)

        # Draw caption info
        gid = rec.get("identity_group_id", rec.get("global_id", "N/A"))
        score = rec.get("similarity_score")
        top1 = rec.get("top1_summary", {})
        gender = top1.get("gender", "")
        upper = top1.get("upper_color", "")
        lower = top1.get("lower_color", "")
        n_tracks = rec.get("matched_track_count", 1)

        caption_y = y + crop_h + 5
        gid_disp = gid if len(gid) <= 24 else gid[:21] + "..."
        draw.text((x + 5, caption_y), gid_disp, fill=(0, 220, 255), font=font)

        if score is not None:
            draw.text((x + 5, caption_y + 16), f"Sim: {score*100:.1f}% | {gender} | {upper}", fill=(255, 215, 0), font=font)
        else:
            draw.text((x + 5, caption_y + 16), f"{gender} | Up: {upper} | Low: {lower}", fill=(220, 220, 220), font=font)

        if n_tracks > 1:
            draw.text((x + 5, caption_y + 32), f"Tracks: {n_tracks} (occluded)", fill=(180, 255, 180), font=font)

    canvas.save(out_path)
    return str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def parse_args():
    parser = argparse.ArgumentParser(description="Person Retrieval Query Engine (Level 5 Part B)")
    # Attribute filters
    parser.add_argument("--gender", type=str, default=None, help="Filter by gender (Female, Male)")
    parser.add_argument("--age", type=str, default=None, help="Filter by age (Young, Adult, Old)")
    parser.add_argument("--hair", type=str, default=None, help="Filter by hair (Short, Long, Bald)")
    parser.add_argument("--glasses", type=str, default=None, help="Filter by glasses (Glasses, Sunglasses, None)")
    parser.add_argument("--hat", type=str, default=None, help="Filter by hat (Hat, No Hat)")
    parser.add_argument("--upper_color", type=str, default=None, help="Filter by upper body color (Red, Blue, Black, Yellow, etc.)")
    parser.add_argument("--lower_color", type=str, default=None, help="Filter by lower body color (Blue, Black, Grey, etc.)")
    parser.add_argument("--lower_type", type=str, default=None, help="Filter by lower type (Trousers&Shorts, Skirt&Dress)")
    parser.add_argument("--bag", type=str, default=None, help="Filter by bag (Bag, Backpack, None)")

    # Re-ID Image Query
    parser.add_argument("--query-image", type=str, default=None, help="Path to sample image crop for Re-ID similarity search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidates to return")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH, help="Path to OSNet checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("LEVEL 5: PERSON RETRIEVAL QUERY ENGINE (PART B)")
    print("=" * 80)

    # Trường hợp biên 2: Người dùng không cung cấp bất kỳ điều kiện tìm kiếm nào
    active_conds = get_active_conditions(args)
    if not active_conds:
        print("\n[LỖI] Cần ít nhất 1 điều kiện tìm kiếm (thuộc tính UPAR hoặc --query-image). Ví dụ: --gender female --upper_color red")
        sys.exit(1)

    # Trường hợp biên 3: Kiểm tra tệp ảnh query_image trước khi nạp CSDL
    if args.query_image:
        img_p = Path(args.query_image)
        if not img_p.exists():
            print(f"\n[LỖI] Tệp ảnh query-image không tồn tại tại: '{args.query_image}'")
            sys.exit(1)

    db = load_database()
    print(f"[INFO] Đã nạp CSDL người đi bộ gồm {len(db)} bản ghi track.")

    # Bước 1: Attribute Filtering (Lọc thuộc tính)
    filtered_recs = filter_by_attributes(db, args)
    print(f"[INFO] Số bản ghi khớp điều kiện thuộc tính: {len(filtered_recs)} track records.")

    # Bước 2: Re-ID Image Ranking (Xếp hạng ảnh mẫu Re-ID nếu có)
    is_reid_query = args.query_image is not None
    if is_reid_query:
        extractor = ReIDExtractor(model_name="osnet_x1_0", checkpoint_path=args.checkpoint)
        ranked_recs = rank_by_reid_similarity(filtered_recs, args.query_image, extractor)
    else:
        ranked_recs = filtered_recs

    # Bước 3: Khử trùng lặp identity bằng identity_group_id
    final_results = deduplicate_by_identity_group(ranked_recs)

    # Lấy top-k đối tượng
    top_results = final_results[:args.top_k]

    # Trường hợp biên 1: Không có kết quả phù hợp (0 results)
    if not top_results:
        print("\n" + "=" * 80)
        print("[THÔNG BÁO] Không tìm thấy cá nhân nào phù hợp với điều kiện tìm kiếm.")
        print("=" * 80)
        print("\nCác điều kiện tìm kiếm đã sử dụng:")
        for k, v in active_conds.items():
            print(f"  - {k}: {v}")

        if args.glasses or args.bag:
            print("\n" + "!" * 80)
            print("Lưu ý: Đầu dự đoán (head) 'glasses' và 'bag' có F1-score thấp khi train (49.32% và 67.15%) do mẫu hiếm trong dataset - có thể có người phù hợp nhưng bị bỏ sót.")
            print("!" * 80)

        return

    # Bước 4: Hiển thị Bảng Kết quả trên Terminal
    print("\n" + "=" * 105)
    print("QUERY RESULTS TABLE (DEDUPLICATED BY GROUND-TRUTH IDENTITY)")
    print("=" * 105)

    if is_reid_query:
        header = f"{'Rank':<5} | {'Global / Group ID':<34} | {'Video Name':<26} | {'Similarity':<10} | {'Attributes (G / Up / Low / Bag)':<25}"
    else:
        header = f"{'STT':<5} | {'Global / Group ID':<34} | {'Video Name':<26} | {'Attributes (G / Up / Low / Bag)':<35}"
    print(header)
    print("-" * 105)

    for idx, rec in enumerate(top_results, 1):
        gid = rec.get("identity_group_id", rec.get("global_id"))
        vname = rec.get("video_name")
        top1 = rec.get("top1_summary", {})
        gender = top1.get("gender", "-")
        upper = top1.get("upper_color", "-")
        lower = top1.get("lower_color", "-")
        bag = top1.get("bag", "-")
        attr_str = f"{gender} | {upper} | {lower} | {bag}"
        note = rec.get("note", "")

        if is_reid_query:
            score = rec.get("similarity_score", 0.0)
            score_str = f"{score*100:6.2f}%"
            print(f"{idx:<5} | {gid:<34} | {vname:<26} | {score_str:<10} | {attr_str:<25} {note}")
        else:
            print(f"{idx:<5} | {gid:<34} | {vname:<26} | {attr_str:<35} {note}")

    print("=" * 105)
    print(f"Tổng số cá nhân độc lập trả về: {len(top_results)} (trên tổng số {len(final_results)} bản ghi phù hợp)")

    # Bước 5: Xuất file ảnh Lưới kết quả (Grid Image)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_filename = f"result_{ts_str}.png"
    grid_path = generate_result_grid_image(top_results, grid_filename)
    if grid_path:
        print(f"\n[OUTPUT GRID IMAGE] Đã lưu file ảnh lưới kết quả tại: '{grid_path}'")

    # Bước 6: Thông báo cảnh báo bắt buộc
    if is_reid_query:
        print("\n" + "!" * 80)
        print("Lưu ý: Độ chính xác Re-ID dao động 5-22% tùy điều kiện camera (xem TECHNICAL_REPORT.md mục Level 3) - kết quả là tín hiệu tham khảo, không phải kết luận chắc chắn.")
        print("!" * 80)

    if (args.glasses or args.bag) and len(top_results) <= 2:
        print("\n" + "!" * 80)
        print("Lưu ý: Đầu dự đoán (head) 'glasses' và 'bag' có F1-score thấp khi train (49.32% và 67.15%) do mẫu hiếm trong dataset - có thể có người phù hợp nhưng bị bỏ sót.")
        print("!" * 80)


if __name__ == "__main__":
    main()
