#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking/demo_combined.py
Generates a single combined demo video demonstrating:
- Level 1: Multi-Object Tracking
- Level 2: Attribute Recognition (11-Heads UPAR Side Info Panel)
- Level 3: Person Re-Identification (Re-ID across track breaks, if GT available)

Canvas output: 960x360 (640x360 main video frame + 320x360 side info panel)

Usage:
    python tracking/demo_combined.py --video-name store-aisle-detection
    python tracking/demo_combined.py --video-name real_pedestrians
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Set encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# Predefined harmonious color palette (RGB)
COLOR_PALETTE = [
    (235, 45, 45),    # Red / Coral
    (255, 140, 20),   # Orange
    (240, 200, 20),   # Yellow/Gold
    (180, 50, 230),   # Purple/Magenta
    (35, 195, 85),    # Green
    (30, 160, 240),   # Cyan/Blue
    (240, 80, 160),   # Pink
    (50, 200, 220),   # Turquoise
    (220, 120, 50),   # Ochre
    (140, 80, 240),   # Violet
    (80, 220, 120),   # Lime
    (230, 150, 200),  # Rose
]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate combined demo video with UPAR side info panel")
    parser.add_argument("--video-name", type=str, default="store-aisle-detection",
                        help="Tên video (vd: 'store-aisle-detection', 'real_pedestrians', 'vtest', 'people-detection')")
    parser.add_argument("--output", type=str, default=None,
                        help="Đường dẫn file output mp4 (mặc định: reports/tracking/demo/demo_combined_v2_full_attributes.mp4)")
    return parser.parse_args()

def get_letter_name(index):
    result = ""
    while True:
        result = chr(65 + (index % 26)) + result
        index = index // 26 - 1
        if index < 0:
            break
    return result

def get_fonts():
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 20)
        font_large = ImageFont.truetype("arialbd.ttf", 16)
        font_medium = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
        font_bold = ImageFont.truetype("arialbd.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    return font_title, font_large, font_medium, font_small, font_bold

def draw_corner_rect(draw, bbox, color, stroke=2, corner_len=9):
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
    
    c_w = stroke + 2
    draw.line([(x1, y1), (x1 + corner_len, y1)], fill=color, width=c_w)
    draw.line([(x1, y1), (x1, y1 + corner_len)], fill=color, width=c_w)
    draw.line([(x2, y1), (x2 - corner_len, y1)], fill=color, width=c_w)
    draw.line([(x2, y1), (x2, y1 + corner_len)], fill=color, width=c_w)
    draw.line([(x1, y2), (x1 + corner_len, y2)], fill=color, width=c_w)
    draw.line([(x1, y2), (x1, y2 - corner_len)], fill=color, width=c_w)
    draw.line([(x2, y2), (x2 - corner_len, y2)], fill=color, width=c_w)
    draw.line([(x2, y2), (x2, y2 - corner_len)], fill=color, width=c_w)

def render_intro_card(w, h, duration_sec, fps, font_title, font_large, font_medium, font_small, video_filename, has_gt):
    frames = []
    total_frames = int(duration_sec * fps)
    
    for _ in range(total_frames):
        img = Image.new('RGB', (w, h), color=(15, 22, 32))
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([0, 0, w, 6], fill=(50, 130, 240))
        
        if has_gt:
            title_text = "Demo: Video Tracking + Attribute Recognition + Person Re-Identification"
            badge_text = "Tự động hợp nhất nhận dạng (L1 + L2 + L3) + UPAR 11-Heads Info Panel"
            
            draw.text((w // 2, h // 2 - 45), title_text, fill=(255, 255, 255), font=font_title, anchor="mm")
            
            sub_text = f"Nguồn video: {video_filename}"
            draw.text((w // 2, h // 2 + 10), sub_text, fill=(180, 200, 220), font=font_large, anchor="mm")
            
            draw.text((w // 2, h // 2 + 50), badge_text, fill=(100, 220, 150), font=font_medium, anchor="mm")
        else:
            title_text = "Demo: Video Tracking + Attribute Recognition"
            note_text = "Lưu ý: Video này không có trường hợp mất track/xuất hiện lại (Re-ID)."
            badge_text = "Tự động nhận dạng & thuộc tính (L1 + L2) + UPAR 11-Heads Info Panel"
            
            draw.text((w // 2, h // 2 - 55), title_text, fill=(255, 255, 255), font=font_title, anchor="mm")
            draw.text((w // 2, h // 2 - 20), note_text, fill=(255, 205, 80), font=font_medium, anchor="mm")
            
            sub_text = f"Nguồn video: {video_filename}"
            draw.text((w // 2, h // 2 + 15), sub_text, fill=(180, 200, 220), font=font_large, anchor="mm")
            
            draw.text((w // 2, h // 2 + 55), badge_text, fill=(100, 220, 150), font=font_medium, anchor="mm")
        
        draw.text((w // 2, h - 25), "Màu sắc = danh tính (identity) | Side Panel = Đầy đủ 11 UPAR Attribute Heads", fill=(140, 160, 180), font=font_small, anchor="mm")
        
        frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    return frames

def draw_info_panel(draw, panel_rect, selected_tid, identity_map, track_summary, is_reid_event, font_large, font_medium, font_small, font_bold):
    px1, py1, px2, py2 = panel_rect
    pw = px2 - px1
    
    # Background
    draw.rectangle([px1, py1, px2, py2], fill=(18, 24, 36))
    
    if selected_tid is not None and selected_tid in identity_map:
        id_info = identity_map[selected_tid]
        identity_name = id_info['name']
        identity_color = id_info['color_rgb']
        
        # Border separating video and panel
        draw.line([(px1, py1), (px1, py2)], fill=identity_color, width=2)

        # Header Box
        header_rect = [px1 + 8, py1 + 8, px2 - 8, py1 + 48]
        draw.rectangle(header_rect, fill=(26, 36, 54), outline=identity_color, width=2)
        
        header_title = f"THÔNG TIN: {identity_name}"
        draw.text((px1 + 16, py1 + 16), header_title, fill=(255, 255, 255), font=font_bold)

        if is_reid_event:
            draw.rectangle([px2 - 110, py1 + 18, px2 - 14, py1 + 38], fill=(180, 50, 50), outline=(255, 215, 0), width=1)
            draw.text((px2 - 62, py1 + 28), "RE-ID EVENT", fill=(255, 255, 255), font=font_small, anchor="mm")
        else:
            draw.text((px2 - 60, py1 + 28), "UPAR 11-HEADS", fill=(140, 160, 180), font=font_small, anchor="mm")

        # Extract UPAR attribute heads
        top1 = track_summary.get(selected_tid, {}).get('summary', {})
        age = top1.get('age', 'N/A')
        age_conf = top1.get('age_conf', 0.0)
        gender = top1.get('gender', 'N/A')
        gender_conf = top1.get('gender_conf', 0.0)
        hair = top1.get('hair', 'N/A')
        hair_conf = top1.get('hair_conf', 0.0)
        glasses = top1.get('glasses', 'None')
        glasses_conf = top1.get('glasses_conf', 0.0)
        hat = top1.get('hat', 'None')
        hat_conf = top1.get('hat_conf', 0.0)
        upper_len = top1.get('upper_length', 'N/A')
        upper_color = top1.get('upper_color', 'N/A')
        lower_len = top1.get('lower_length', 'N/A')
        lower_color = top1.get('lower_color', 'N/A')
        lower_type = top1.get('lower_type', 'N/A')
        bag = top1.get('bag', 'None')

        # Formatted lines
        age_str = f"{age} ({int(round(age_conf * 100))}%)" if age_conf > 0 else f"{age}"
        gender_str = f"{gender} ({int(round(gender_conf * 100))}%)" if gender_conf > 0 else f"{gender}"
        hair_str = f"{hair} ({int(round(hair_conf * 100))}%)" if hair_conf > 0 else f"{hair}"
        glasses_str = f"{glasses}" + (f" ({int(round(glasses_conf * 100))}%)" if glasses != "None" and glasses_conf > 0 else "")
        hat_str = f"{hat}" + (f" ({int(round(hat_conf * 100))}%)" if hat != "No Hat" and hat != "None" and hat_conf > 0 else "")
        upper_str = f"{upper_len} sleeve, {upper_color}".strip(", ")
        lower_str = f"{lower_len}, {lower_color}, {lower_type}".strip(", ")
        bag_str = f"{bag}"

        lines = [
            ("Age:", age_str),
            ("Gender:", gender_str),
            ("Hair:", hair_str),
            ("Glasses:", glasses_str),
            ("Hat:", hat_str),
            ("Upper:", upper_str),
            ("Lower:", lower_str),
            ("Bag:", bag_str),
        ]

        start_y = py1 + 60
        line_height = 32

        for i, (label, val) in enumerate(lines):
            cur_y = start_y + i * line_height
            
            if i % 2 == 1:
                draw.rectangle([px1 + 10, cur_y - 2, px2 - 10, cur_y + 26], fill=(23, 31, 46))
            
            draw.text((px1 + 16, cur_y + 4), label, fill=(160, 185, 210), font=font_bold)
            draw.text((px1 + 90, cur_y + 4), val, fill=(255, 255, 255), font=font_medium)

        draw.text((px1 + pw // 2, py2 - 18), "Identity-level UPAR attribute summary", fill=(100, 120, 145), font=font_small, anchor="mm")

    else:
        # Empty frame state
        draw.line([(px1, py1), (px1, py2)], fill=(60, 75, 90), width=2)
        
        header_rect = [px1 + 8, py1 + 8, px2 - 8, py1 + 48]
        draw.rectangle(header_rect, fill=(26, 36, 54), outline=(70, 85, 100), width=1)
        draw.text((px1 + 16, py1 + 16), "THÔNG TIN: Không có", fill=(180, 190, 200), font=font_bold)

        msg1 = "Khong co doi tuong nao"
        msg2 = "trong khung hinh"
        draw.text((px1 + pw // 2, py1 + 150), msg1, fill=(140, 155, 175), font=font_medium, anchor="mm")
        draw.text((px1 + pw // 2, py1 + 175), msg2, fill=(140, 155, 175), font=font_medium, anchor="mm")

def main():
    args = parse_args()
    video_name = args.video_name

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    mp4_path = os.path.join(BASE_DIR, 'tracking', 'test_videos', f"{video_name}.mp4")
    avi_path = os.path.join(BASE_DIR, 'tracking', 'test_videos', f"{video_name}.avi")
    if os.path.exists(mp4_path):
        video_path = mp4_path
    elif os.path.exists(avi_path):
        video_path = avi_path
    else:
        video_path = mp4_path

    csv_path = os.path.join(BASE_DIR, 'reports', 'tracking', video_name, 'tracks.csv')
    attr_path = os.path.join(BASE_DIR, 'reports', 'tracking', video_name, 'attributes.json')
    gt_path = os.path.join(BASE_DIR, 'reports', 'tracking', video_name, 'reentry_ground_truth.csv')
    
    output_dir = os.path.join(BASE_DIR, 'reports', 'tracking', 'demo')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.output:
        output_video_path = args.output
    else:
        if video_name == "store-aisle-detection":
            output_video_path = os.path.join(output_dir, "demo_combined_v2_full_attributes.mp4")
        else:
            output_video_path = os.path.join(output_dir, f"demo_{video_name}_v2.mp4")

    print(f"=== STARTING DEMO VIDEO GENERATION (V2 FULL ATTRIBUTES) FOR '{video_name}' ===")
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file tracks CSV: {csv_path}")
    print(f"Reading tracks CSV: {csv_path}")
    df_tracks = pd.read_csv(csv_path)

    track_summary = {}
    if os.path.exists(attr_path):
        print(f"Reading attributes JSON: {attr_path}")
        with open(attr_path, 'r', encoding='utf-8') as f:
            attr_data = json.load(f)
        for tid_str, data in attr_data.items():
            tid = int(tid_str)
            track_summary[tid] = {
                "first_seen_frame": data.get("first_seen_frame", 0),
                "last_seen_frame": data.get("last_seen_frame", 0),
                "first_seen_time": data.get("first_seen_time", 0.0),
                "last_seen_time": data.get("last_seen_time", 0.0),
                "summary": data.get("top1_summary", {})
            }
    else:
        print(f"Warning: attributes.json not found at {attr_path}")

    has_gt = False
    df_gt = None
    if os.path.exists(gt_path):
        try:
            df_gt = pd.read_csv(gt_path)
            if not df_gt.empty and 'track_id_a' in df_gt.columns and 'track_id_b' in df_gt.columns:
                has_gt = True
        except Exception:
            has_gt = False

    identity_map = {}
    prev_track_map = {}
    
    all_track_ids = sorted(df_tracks['track_id'].unique())

    if has_gt:
        print(f"Reading ground truth CSV: {gt_path}")
        parent = {tid: tid for tid in all_track_ids}
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        for _, row in df_gt.iterrows():
            ta = int(row['track_id_a'])
            tb = int(row['track_id_b'])
            if ta in parent and tb in parent:
                root_a = find(ta)
                root_b = find(tb)
                if root_a != root_b:
                    if root_a < root_b:
                        parent[root_b] = root_a
                    else:
                        parent[root_a] = root_b

        clusters = {}
        for tid in all_track_ids:
            root = find(tid)
            clusters.setdefault(root, []).append(tid)

        sorted_roots = sorted(clusters.keys())

        for idx, root in enumerate(sorted_roots):
            group_tids = clusters[root]
            group_tids.sort(key=lambda t: track_summary.get(t, {}).get("first_seen_frame", t))
            
            letter_name = get_letter_name(idx)
            person_id = f"Person {letter_name}"
            color_rgb = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

            for tid in group_tids:
                identity_map[tid] = {
                    "id": person_id,
                    "name": f"{person_id}",
                    "color_rgb": color_rgb
                }

            for i in range(1, len(group_tids)):
                prev_t = group_tids[i - 1]
                cur_t = group_tids[i]
                prev_track_map[cur_t] = prev_t
    else:
        print("[CANH BAO] Video nay khong co reentry_ground_truth.csv - demo se chi minh hoa Level 1+2 (Tracking + Attribute), khong co minh hoa Re-ID.")
        for idx, tid in enumerate(all_track_ids):
            color_rgb = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            identity_map[tid] = {
                "id": f"ID {tid}",
                "name": f"ID {tid}",
                "color_rgb": color_rgb
            }

    reid_banners = {}
    for tid, prev_tid in prev_track_map.items():
        if tid in track_summary and prev_tid in track_summary:
            t_start = track_summary[tid]["first_seen_frame"]
            cur_time = track_summary[tid]["first_seen_time"]
            prev_time = track_summary[prev_tid]["last_seen_time"]
            gap_sec = max(0.1, cur_time - prev_time)
            msg = f"<< Re-ID: nhận lại từ track {prev_tid}, cách {gap_sec:.1f}s >>"
            reid_banners[tid] = {
                "start_frame": t_start,
                "end_frame": t_start + 90,
                "msg": msg
            }

    # 2. Open Original Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Target resolution: Video frame 640x360 + Panel width 320 -> Canvas 960x360
    v_w, v_h = 640, 360
    panel_w = 320
    out_w, out_h = v_w + panel_w, v_h
    scale_x, scale_y = v_w / orig_w, v_h / orig_h

    if orig_fps > 45.0:
        output_fps = 30.0
        sample_step = 2
    else:
        output_fps = float(orig_fps)
        sample_step = 1

    print(f"Video input: {orig_w}x{orig_h}, {orig_fps:.2f} FPS, {total_frames_orig} frames")
    print(f"Target canvas output: {out_w}x{out_h} (Video: 640x360 + Panel: 320x360), {output_fps:.2f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (out_w, out_h))
    
    font_title, font_large, font_medium, font_small, font_bold = get_fonts()

    # 4. Render Intro Card (3.0 seconds)
    print("Rendering 3-second Intro Card...")
    video_filename = os.path.basename(video_path)
    intro_frames = render_intro_card(out_w, out_h, duration_sec=3.0, fps=output_fps, font_title=font_title, font_large=font_large, font_medium=font_medium, font_small=font_small, video_filename=video_filename, has_gt=has_gt)
    for frame in intro_frames:
        out.write(frame)

    # 5. Process Main Video Frames
    print("Processing main video frames...")
    
    screenshot_targets = {}
    if video_name == "store-aisle-detection":
        screenshot_targets = {
            725: os.path.join(output_dir, "v2_screenshot_1_frame725_reid_personB.jpg"),
            1575: os.path.join(output_dir, "v2_screenshot_2_frame1575_reid_personA.jpg"),
            2000: os.path.join(output_dir, "v2_screenshot_3_frame2000_normal_personD.jpg"),
            2885: os.path.join(output_dir, "v2_screenshot_4_frame2885_reid_personC.jpg")
        }
    extracted_screenshots = {}

    frame_idx = 0
    written_frames_count = len(intro_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize original video frame to 640x360
        frame_v = cv2.resize(frame, (v_w, v_h), interpolation=cv2.INTER_AREA)

        # Trimming & Speed Up Logic for store-aisle-detection
        if video_name == "store-aisle-detection":
            if frame_idx < 539:
                if frame_idx % 6 != 0:
                    frame_idx += 1
                    continue
            else:
                if (frame_idx % 2 != 0) and (frame_idx not in screenshot_targets):
                    frame_idx += 1
                    continue
        else:
            if sample_step > 1 and (frame_idx % sample_step != 0) and (frame_idx not in screenshot_targets):
                frame_idx += 1
                continue

        # Create 960x360 Canvas
        canvas_img = Image.new('RGB', (out_w, out_h), color=(15, 22, 32))
        
        # Paste video frame on left side (0, 0)
        img_v_rgb = cv2.cvtColor(frame_v, cv2.COLOR_BGR2RGB)
        canvas_img.paste(Image.fromarray(img_v_rgb), (0, 0))

        draw = ImageDraw.Draw(canvas_img, 'RGBA')

        # Get active tracks for current frame
        frame_tracks = df_tracks[df_tracks['frame_id'] == frame_idx]
        
        active_identities = set()
        reid_active_track_in_frame = None
        largest_track_in_frame = None
        max_bbox_area = -1.0

        for _, row in frame_tracks.iterrows():
            tid = int(row['track_id'])
            if tid not in identity_map:
                continue
            
            id_info = identity_map[tid]
            identity_id = id_info['id']
            identity_name = id_info['name']
            color = id_info['color_rgb']
            
            active_identities.add(identity_id)

            # Scale bounding box coordinates to 640x360
            x1, y1, x2, y2 = float(row['x1']) * scale_x, float(row['y1']) * scale_y, float(row['x2']) * scale_x, float(row['y2']) * scale_y
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            # Track largest bbox
            area = (x2 - x1) * (y2 - y1)
            if area > max_bbox_area:
                max_bbox_area = area
                largest_track_in_frame = tid

            # Draw Bbox on video
            draw_corner_rect(draw, bbox, color, stroke=2, corner_len=9)

            # Get Attribute text for short 2-line label
            top1 = track_summary.get(tid, {}).get('summary', {})
            gender = top1.get('gender', '')
            upper_color = top1.get('upper_color', '')
            upper_len = top1.get('upper_length', '')

            line1_parts = []
            if gender:
                line1_parts.append(gender)
            attr_str = f"{upper_color} {upper_len}".strip()
            if attr_str:
                line1_parts.append(attr_str)
            
            attr_text = ", ".join(line1_parts) if line1_parts else "N/A"
            line1_str = f"{identity_id} | {attr_text}"

            # Check if Line 2 (Re-ID banner) should be active
            line2_str = None
            if tid in reid_banners:
                b_info = reid_banners[tid]
                if b_info['start_frame'] <= frame_idx <= b_info['end_frame']:
                    line2_str = b_info['msg']
                    reid_active_track_in_frame = tid

            # Calculate Label Box position above bbox
            lbl_x = bbox[0]
            lbl_y = bbox[1] - 8

            bbox_l1 = font_bold.getbbox(line1_str)
            w_l1 = bbox_l1[2] - bbox_l1[0]
            h_l1 = bbox_l1[3] - bbox_l1[1]

            w_total = w_l1 + 14
            h_total = h_l1 + 8

            if line2_str:
                bbox_l2 = font_small.getbbox(line2_str)
                w_l2 = bbox_l2[2] - bbox_l2[0]
                h_l2 = bbox_l2[3] - bbox_l2[1]
                w_total = max(w_total, w_l2 + 14)
                h_total += h_l2 + 6

            box_top = max(8, lbl_y - h_total)
            box_rect = [lbl_x, box_top, lbl_x + w_total, box_top + h_total]

            draw.rectangle(box_rect, fill=(15, 22, 32, 220), outline=color, width=1)
            draw.text((lbl_x + 7, box_top + 4), line1_str, fill=(255, 255, 255), font=font_bold)

            if line2_str:
                draw.text((lbl_x + 7, box_top + 4 + h_l1 + 4), line2_str, fill=(255, 215, 0), font=font_small)

        # 6. Top-Left HUD Badge: "Active identities: N"
        hud_text = f"Active identities: {len(active_identities)}"
        draw.rectangle([12, 12, 210, 42], fill=(15, 22, 32, 220), outline=(50, 150, 240), width=1)
        draw.ellipse([20, 23, 28, 31], fill=(35, 225, 100))
        draw.text((35, 19), hud_text, fill=(255, 255, 255), font=font_bold)

        # 7. Bottom Watermark Banner
        bm_text = "Màu sắc = danh tính (identity), không phải track ID thô"
        draw.rectangle([12, v_h - 32, v_w - 12, v_h - 8], fill=(15, 22, 32, 190))
        draw.text((v_w // 2, v_h - 20), bm_text, fill=(220, 230, 240), font=font_small, anchor="mm")

        # 8. Selection Rule for UPAR Side Info Panel:
        # Priority 1: Re-ID active track
        # Priority 2: Largest bbox track in frame
        if reid_active_track_in_frame is not None:
            selected_tid = reid_active_track_in_frame
            is_reid_event = True
        elif largest_track_in_frame is not None:
            selected_tid = largest_track_in_frame
            is_reid_event = False
        else:
            selected_tid = None
            is_reid_event = False

        # Draw Side Info Panel (640..960, 0..360)
        draw_info_panel(
            draw=draw,
            panel_rect=[640, 0, 960, 360],
            selected_tid=selected_tid,
            identity_map=identity_map,
            track_summary=track_summary,
            is_reid_event=is_reid_event,
            font_large=font_large,
            font_medium=font_medium,
            font_small=font_small,
            font_bold=font_bold
        )

        # Convert back to OpenCV BGR
        final_frame = cv2.cvtColor(np.array(canvas_img), cv2.COLOR_RGB2BGR)
        out.write(final_frame)
        written_frames_count += 1

        # Check screenshot capture
        if frame_idx in screenshot_targets:
            save_path = screenshot_targets[frame_idx]
            cv2.imwrite(save_path, final_frame)
            time_sec = frame_idx / orig_fps
            extracted_screenshots[frame_idx] = {
                "path": save_path,
                "timestamp_sec": time_sec,
                "timestamp_str": f"{int(time_sec//60):02d}:{time_sec%60:04.1f}"
            }
            print(f"Captured screenshot frame {frame_idx} ({time_sec:.1f}s) -> {os.path.basename(save_path)}")

        frame_idx += 1

    # Cleanup
    cap.release()
    out.release()
    print("Video encoding complete!")

    # 9. Post-processing Verification & Bitrate Check
    final_duration = written_frames_count / output_fps
    file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
    
    print("\n=== OUTPUT METRICS ===")
    print(f"File Path: {output_video_path}")
    print(f"Final Duration: {final_duration:.2f} seconds ({written_frames_count} frames)")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Output FPS: {output_fps:.2f}")

    if extracted_screenshots:
        print("\n=== EXTRACTED SCREENSHOTS ===")
        for f_idx, info in extracted_screenshots.items():
            print(f"- Frame {f_idx:4d} (Timestamp {info['timestamp_str']}): {info['path']}")

if __name__ == "__main__":
    main()
