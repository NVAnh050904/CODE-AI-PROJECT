"""
tracking/track.py
==================
Baseline pipeline: YOLOv8 (person detection) + ByteTrack (tracking).

Muc tieu (Level 1 trong roadmap):
    Video / Webcam --> YOLO person detection --> ByteTrack --> track_id + bbox
    --> luu ket qua structured (CSV) de cac module sau (Attribute Recognition,
    Re-ID) doc lai ma khong phai viet lai pipeline.

Khong train tracker, khong train detector rieng o buoc nay - chi dung
pretrained YOLOv8 (COCO, class 'person' = 0) + ByteTrack co san trong
ultralytics.

Cach chay:
    # Video file
    python tracking/track.py --source path/to/video.mp4

    # Webcam (mac dinh camera index 0)
    python tracking/track.py --source 0

    # Tuy chinh model / tracker / threshold
    python tracking/track.py --source video.mp4 --model yolov8s.pt \
        --tracker bytetrack.yaml --conf 0.4 --save-video

Output:
    reports/tracking/<ten_video>_tracks.csv
        frame_id,track_id,x1,y1,x2,y2,confidence,timestamp
    reports/tracking/<ten_video>_tracked.mp4   (neu --save-video)
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PERSON_CLASS_ID = 0  # COCO class id cho 'person'


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack person tracking baseline")
    parser.add_argument("--source", type=str, required=True,
                         help="Duong dan video (.mp4, .avi, ...) hoac index webcam (vd: 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Weights YOLOv8 dung de detect person (mac dinh: yolov8n.pt, pretrained COCO)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                         choices=["bytetrack.yaml", "botsort.yaml"],
                         help="Tracker config co san trong ultralytics")
    parser.add_argument("--conf", type=float, default=0.35,
                         help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                         help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Kich thuoc anh dua vao model")
    parser.add_argument("--output-dir", type=str, default="reports/tracking",
                         help="Thu muc luu CSV / video ket qua")
    parser.add_argument("--save-video", action="store_true",
                         help="Neu bat, luu lai video co ve bbox + track_id")
    parser.add_argument("--show", action="store_true",
                         help="Hien thi cua so preview khi chay (tat mac dinh, hop ly khi chay tren server)")
    parser.add_argument("--device", type=str, default=None,
                         help="'cpu', '0' (GPU 0), hoac de trong de tu dong chon")
    return parser.parse_args()


def resolve_source(source: str):
    """Cho phep truyen webcam index (vd '0') hoac duong dan file video."""
    if source.isdigit():
        return int(source)
    return source


def get_video_writer(output_path: Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def main():
    args = parse_args()

    source = resolve_source(args.source)
    is_webcam = isinstance(source, int)

    # ten dung de dat ten file output (video file -> ten file; webcam -> "webcam0")
    if is_webcam:
        run_name = f"webcam{source}"
    else:
        run_name = Path(source).stem

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_tracks.csv"
    video_out_path = output_dir / f"{run_name}_tracked.mp4"

    model_path = args.model
    if not Path(model_path).exists() and Path("checkpoints") / model_path:
        ckpt_candidate = Path("checkpoints") / model_path
        if ckpt_candidate.exists():
            model_path = str(ckpt_candidate)

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # Mo capture rieng chi de lay fps/width/height khi can ghi video output.
    # Viec detect + track thuc su se dung model.track(..., stream=True) o duoi.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if args.save_video:
        writer = get_video_writer(video_out_path, fps, width, height)

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence", "timestamp"])

    print(f"[INFO] Bat dau tracking tren nguon: {args.source}")
    print(f"[INFO] Ket qua CSV se luu tai: {csv_path}")
    if args.save_video:
        print(f"[INFO] Video ket qua se luu tai: {video_out_path}")

    frame_id = 0
    id_switch_tracker = {}  # ghi nho track_id -> vi tri gan nhat, chi de log canh bao don gian
    start_time = time.time()

    # model.track voi persist=True se tu duy tri track_id qua cac frame lien tiep.
    results_stream = model.track(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=[PERSON_CLASS_ID],
        tracker=args.tracker,
        device=args.device,
        stream=True,
        persist=True,
        verbose=False,
    )

    for result in results_stream:
        frame_id += 1
        frame = result.orig_img
        timestamp = frame_id / fps

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), track_id, conf in zip(xyxy, track_ids, confs):
                csv_writer.writerow([
                    frame_id, int(track_id),
                    round(float(x1), 2), round(float(y1), 2),
                    round(float(x2), 2), round(float(y2), 2),
                    round(float(conf), 4), round(timestamp, 3),
                ])

                if args.save_video or args.show:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID {int(track_id)} ({conf:.2f})"
                    cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if args.save_video and writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Nguoi dung dung (nhan 'q').")
                break

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] Da xu ly {frame_id} frame ({frame_id / elapsed:.1f} FPS)")

    csv_file.close()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"[DONE] Hoan tat: {frame_id} frame trong {elapsed:.1f}s ({frame_id / max(elapsed, 1e-6):.1f} FPS)")
    print(f"[DONE] CSV: {csv_path}")
    if args.save_video:
        print(f"[DONE] Video: {video_out_path}")


if __name__ == "__main__":
    main()
