"""
tracking/build_person_database.py
==================================
Level 5 - Person Retrieval (Part A): Consolidated Person Database Builder.

Builds and updates a central database of all official pedestrian tracks across videos (reports/tracking/person_database.json).
Supports:
1. Ground-truth identity grouping using networkx (when reentry_ground_truth.csv exists).
2. Representative crop selection (middle frame of each track).
3. Explicit list of 5 official videos (OFFICIAL_VIDEOS).
4. Full rebuild (--rebuild-all or default) and incremental video update (--add-video <name>).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import networkx as nx

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRACKING_DIR = PROJECT_ROOT / "reports" / "tracking"
CROPS_BASE = TRACKING_DIR / "crops"
DATABASE_PATH = TRACKING_DIR / "person_database.json"

# Explicit list of official benchmark videos (4 truly independent domains)
OFFICIAL_VIDEOS = [
    "real_pedestrians",
    "store-aisle-detection",
    "person-bicycle-car-detection",
    "vtest"
]

SYSTEM_DIRS = {
    "crops", "demo", "query_results", "_debug_images", "_exploration_archive"
}


def find_crop_folder_for_video(video_name: str) -> Path:
    """Find crop folder matching video name in reports/tracking/crops."""
    if not CROPS_BASE.exists():
        return None

    candidates = [
        video_name,
        video_name.replace("-", "_"),
        video_name.replace("_", "-"),
        video_name.replace("-detection", "").replace("-", "_"),
        video_name.replace("_detection", "").replace("_", "-"),
    ]
    for c in candidates:
        p = CROPS_BASE / c
        if p.exists() and p.is_dir():
            return p
    
    # Fuzzy search subdirectories in crops folder
    for d in CROPS_BASE.iterdir():
        if d.is_dir() and any(c in d.name for c in candidates):
            return d
            
    return None


def get_representative_crop(crop_base_dir: Path, track_id: int) -> str:
    """Select the crop image in the middle of the track."""
    if crop_base_dir is None or not crop_base_dir.exists():
        return ""

    track_folder = crop_base_dir / f"track_{track_id}"
    if not track_folder.exists() or not track_folder.is_dir():
        return ""

    crop_files = sorted(
        [f for f in track_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=lambda f: (
            int(f.stem.split("_")[1]) if len(f.stem.split("_")) > 1 and f.stem.split("_")[1].isdigit() else f.name
        )
    )

    if not crop_files:
        return ""

    middle_crop = crop_files[len(crop_files) // 2]
    try:
        rel_path = middle_crop.relative_to(PROJECT_ROOT)
        return str(rel_path).replace("\\", "/")
    except ValueError:
        return str(middle_crop).replace("\\", "/")


def build_identity_groups(video_dir: Path, all_track_ids: list) -> dict:
    """
    Build ground-truth identity groups using networkx if reentry_ground_truth.csv exists.
    Returns dict: track_id -> { "identity_group_id": str, "linked_global_ids": list[str] }
    """
    video_name = video_dir.name
    gt_csv = video_dir / "reentry_ground_truth.csv"

    group_mapping = {}

    if gt_csv.exists() and gt_csv.is_file():
        try:
            df = pd.read_csv(gt_csv)
            G = nx.Graph()
            for t in all_track_ids:
                G.add_node(t)
            for _, row in df.iterrows():
                a, b = int(row["track_id_a"]), int(row["track_id_b"])
                if a != b:
                    G.add_edge(a, b)

            components = sorted([sorted(list(c)) for c in nx.connected_components(G)], key=lambda x: x[0])

            for comp in components:
                min_t = comp[0]
                if len(comp) > 1:
                    group_id = f"{video_name}::Group_{min_t}"
                    all_globals = [f"{video_name}::{t}" for t in comp]
                    for t in comp:
                        g_id = f"{video_name}::{t}"
                        linked = [gid for gid in all_globals if gid != g_id]
                        group_mapping[t] = {
                            "identity_group_id": group_id,
                            "linked_global_ids": linked
                        }
                else:
                    g_id = f"{video_name}::{min_t}"
                    group_mapping[min_t] = {
                        "identity_group_id": g_id,
                        "linked_global_ids": []
                    }
        except Exception as e:
            print(f"[WARNING] Failed to parse {gt_csv}: {e}. Falling back to singletons.")
            for t in all_track_ids:
                g_id = f"{video_name}::{t}"
                group_mapping[t] = {"identity_group_id": g_id, "linked_global_ids": []}
    else:
        for t in all_track_ids:
            g_id = f"{video_name}::{t}"
            group_mapping[t] = {"identity_group_id": g_id, "linked_global_ids": []}

    return group_mapping


def process_video_directory(video_dir: Path) -> list:
    """Read attributes.json for a single video directory and convert to record list."""
    attr_json = video_dir / "attributes.json"
    if not attr_json.exists():
        return []

    video_name = video_dir.name
    crop_folder = find_crop_folder_for_video(video_name)

    with open(attr_json, "r", encoding="utf-8") as f:
        attr_data = json.load(f)

    all_track_ids = [int(k) for k in attr_data.keys() if k.isdigit()]
    identity_groups = build_identity_groups(video_dir, all_track_ids)

    records = []
    for k_str, track_info in attr_data.items():
        if not k_str.isdigit():
            continue
        track_id = int(k_str)
        global_id = f"{video_name}::{track_id}"

        group_info = identity_groups.get(track_id, {
            "identity_group_id": global_id,
            "linked_global_ids": []
        })

        crop_path = get_representative_crop(crop_folder, track_id)

        rec = {
            "global_id": global_id,
            "identity_group_id": group_info["identity_group_id"],
            "linked_global_ids": group_info["linked_global_ids"],
            "video_name": video_name,
            "track_id": track_id,
            "first_seen_frame": track_info.get("first_seen_frame", 0),
            "last_seen_frame": track_info.get("last_seen_frame", 0),
            "first_seen_time": track_info.get("first_seen_time", 0.0),
            "last_seen_time": track_info.get("last_seen_time", 0.0),
            "top1_summary": track_info.get("top1_summary", {}),
            "multi_label_heads": track_info.get("multi_label_heads", {}),
            "embedding": track_info.get("embedding", []),
            "representative_crop": crop_path
        }
        records.append(rec)

    return records


def rebuild_all_database() -> dict:
    """Read attributes.json ONLY for the 5 official videos in OFFICIAL_VIDEOS."""
    print("[INFO] Rebuilding person database from scratch for OFFICIAL_VIDEOS...")

    # Check for unlisted video directories that contain attributes.json
    unlisted = []
    for d in TRACKING_DIR.iterdir():
        if d.is_dir() and d.name not in SYSTEM_DIRS and d.name not in OFFICIAL_VIDEOS:
            if (d / "attributes.json").exists():
                unlisted.append(d.name)

    if unlisted:
        print("\n" + "!" * 75)
        print(f"[CANH BAO] Phat hien các thu muc co attributes.json nhung KHONG nam trong OFFICIAL_VIDEOS:")
        for u in unlisted:
            print(f"  - {u}")
        print("Cac video nay se KHONG duoc dua vao person_database.json.")
        print("!" * 75 + "\n")

    database = {}
    for v_name in OFFICIAL_VIDEOS:
        v_dir = TRACKING_DIR / v_name
        if v_dir.exists() and (v_dir / "attributes.json").exists():
            records = process_video_directory(v_dir)
            for r in records:
                database[r["global_id"]] = r
            print(f"  - Loaded {len(records):2d} tracks from official video '{v_name}'")
        else:
            print(f"  - [WARNING] Official video directory '{v_name}' or attributes.json not found!")

    save_database(database)
    return database


def save_database(database: dict):
    """Save database dictionary to reports/tracking/person_database.json."""
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATABASE_PATH, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2)
    print(f"[OUTPUT] Person database saved to '{DATABASE_PATH.resolve()}'")


def load_database() -> dict:
    """Load database dictionary from file if exists, else return None."""
    if DATABASE_PATH.exists():
        try:
            with open(DATABASE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read database file: {e}")
            return None
    return None


def add_video_to_database(video_name: str):
    """
    Add or update video in person_database.json.
    Rejects videos that are not in OFFICIAL_VIDEOS.
    """
    if video_name not in OFFICIAL_VIDEOS:
        print("\n" + "!" * 80)
        print(f"[TU CHOI THUC HIEN] Video '{video_name}' khong thuoc nhom chinh thuc")
        print("(xem reports/tracking/_exploration_archive/README.md).")
        print("Neu muon dua vao database chinh thuc, can xac nhan lai ground-truth truoc.")
        print("!" * 80 + "\n")
        sys.exit(1)

    database = load_database()
    if database is None:
        print(f"[NOTICE] Database not found. Rebuilding all first...")
        database = rebuild_all_database()

    video_dir = TRACKING_DIR / video_name
    if not video_dir.exists() or not (video_dir / "attributes.json").exists():
        raise FileNotFoundError(f"[ERROR] Video directory or attributes.json not found at '{video_dir}'")

    count_before = len(database)

    # Remove existing records for this video to ensure overwrite logic
    filtered_db = {gid: rec for gid, rec in database.items() if rec.get("video_name") != video_name}

    new_records = process_video_directory(video_dir)
    for r in new_records:
        filtered_db[r["global_id"]] = r

    count_after = len(filtered_db)
    save_database(filtered_db)

    existing_videos = sorted(list(set(r.get("video_name") for r in filtered_db.values())))

    print("\n" + "=" * 65)
    print("INCREMENTAL VIDEO ADDITION SUMMARY")
    print("=" * 65)
    print(f"Target Video           : {video_name}")
    print(f"Total Records Before   : {count_before}")
    print(f"Total Records After    : {count_after}")
    print(f"Videos in Database ({len(existing_videos)}) : {', '.join(existing_videos)}")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidated Person Database Builder (Level 5 Part A)")
    parser.add_argument("--rebuild-all", action="store_true", help="Build person_database.json from scratch for OFFICIAL_VIDEOS")
    parser.add_argument("--add-video", type=str, default=None, help="Add or update a specific official video's records in person_database.json")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("LEVEL 5: PERSON DATABASE BUILDER (PART A)")
    print("=" * 65)

    if args.add_video:
        add_video_to_database(args.add_video)
    else:
        database = rebuild_all_database()
        existing_videos = sorted(list(set(r.get("video_name") for r in database.values())))
        print("\n" + "=" * 65)
        print("DATABASE BUILD SUMMARY")
        print("=" * 65)
        print(f"Total Records Built    : {len(database)}")
        print(f"Videos in Database ({len(existing_videos)}) : {', '.join(existing_videos)}")
        print("=" * 65)


if __name__ == "__main__":
    main()
