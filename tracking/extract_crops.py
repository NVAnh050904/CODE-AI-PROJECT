"""
tracking/extract_crops.py
==========================
Buoc chuan bi cho Level 2 (Attribute tracking) trong roadmap.

Doc file CSV do track.py sinh ra, cat crop nguoi tu video theo tung
(frame_id, track_id, bbox) va luu thanh anh rieng. Cac crop nay sau do
co the dua thang vao pipeline UPAR hien co (predict_image.py) de gan
attribute cho tung track_id.

Cach chay:
    python tracking/extract_crops.py \
        --video path/to/video.mp4 \
        --csv reports/tracking/video_tracks.csv \
        --output-dir reports/tracking/crops \
        --every-n-frames 5

--every-n-frames: khong can luu crop cho MOI frame (du thua, ton dung
luong). Vi du 5 nghia la cu 5 frame lay 1 crop cho moi track_id.

Output:
    reports/tracking/crops/track_<track_id>/frame_<frame_id>.jpg
"""

import argparse
import csv as csv_module
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract person crops from tracking CSV")
    parser.add_argument("--video", type=str, required=True, help="Video goc (cung file da dung de tracking)")
    parser.add_argument("--csv", type=str, required=True, help="File CSV ket qua tu track.py")
    parser.add_argument("--output-dir", type=str, default="reports/tracking/crops")
    parser.add_argument("--every-n-frames", type=int, default=5,
                         help="Chi luu 1 crop moi N frame cho tung track_id")
    parser.add_argument("--padding", type=float, default=0.0,
                         help="Ty le padding them quanh bbox (0.1 = them 10%% moi chieu)")
    parser.add_argument("--clean", action="store_true",
                         help="Xoa thu muc output-dir cu truoc khi xuat crop moi (tranh lan crop cu)")
    return parser.parse_args()


def load_tracks(csv_path: str):
    """Tra ve dict: frame_id -> list[(track_id, x1, y1, x2, y2, conf)]"""
    tracks_by_frame = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            tracks_by_frame[frame_id].append((
                int(row["track_id"]),
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
                float(row["confidence"]),
            ))
    return tracks_by_frame


def pad_box(x1, y1, x2, y2, padding, frame_w, frame_h):
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * padding)
    y1 = max(0, y1 - h * padding)
    x2 = min(frame_w, x2 + w * padding)
    y2 = min(frame_h, y2 + h * padding)
    return x1, y1, x2, y2


def main():
    args = parse_args()

    tracks_by_frame = load_tracks(args.csv)
    print(f"[INFO] Da doc {len(tracks_by_frame)} frame co track tu CSV: {args.csv}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {args.video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print(f"[INFO] Da xoa thu muc crop cu: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # dem so crop da luu cho moi track_id de ap dung --every-n-frames
    saved_count = defaultdict(int)
    saved_total = 0

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        entries = tracks_by_frame.get(frame_id)
        if not entries:
            continue

        for track_id, x1, y1, x2, y2, conf in entries:
            saved_count[track_id] += 1
            if saved_count[track_id] % args.every_n_frames != 0:
                continue

            x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, args.padding, frame_w, frame_h)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            track_dir = output_dir / f"track_{track_id}"
            track_dir.mkdir(parents=True, exist_ok=True)
            out_path = track_dir / f"frame_{frame_id}.jpg"
            cv2.imwrite(str(out_path), crop)
            saved_total += 1

    cap.release()
    print(f"[DONE] Da luu {saved_total} crop vao: {output_dir}")


if __name__ == "__main__":
    main()
