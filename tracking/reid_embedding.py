"""
tracking/reid_embedding.py
===========================
Level 3 in roadmap: Person Re-Identification (Re-ID) Feature Embedding Extractor.

Features:
1. ReIDExtractor: Pretrained OSNet model loader (512-dim embedding extraction).
2. Market1501 Validation: Evaluates Intra-ID vs Inter-ID similarity distributions on Market1501 dataset (20 random identities).
3. 5x5 Identity Cosine Similarity Matrix display for Market1501.
4. Track Aggregation: Computes mean-pooled L2-normalized 512-dim embedding for each track, updates reports/tracking/track_attributes.json, and prints 7x7 track similarity matrix.
"""

import os
import sys
import json
import random
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchreid

# Pretrained OSNet weights URL (MSMT17 trained, 512-dim output)
PRETRAINED_MODEL_URL = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
DEFAULT_CHECKPOINT_PATH = "checkpoints/osnet_x1_0_msmt17.pth"


class ReIDExtractor:
    """
    Feature Extractor using pretrained OSNet model.
    Input: image path or list of image paths.
    Output: L2-normalized 512-dim embedding vector(s) as np.ndarray.
    """
    def __init__(self, model_name: str = "osnet_x1_0", checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            print(f"[INFO] Pretrained Re-ID checkpoint not found at '{checkpoint_path}'. Downloading...")
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(PRETRAINED_MODEL_URL, str(ckpt_file))
            print(f"[INFO] Downloaded weights ({ckpt_file.stat().st_size} bytes) to '{checkpoint_path}'")

        print(f"[INFO] Loading Re-ID model '{model_name}' on device '{self.device}'...")
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=str(ckpt_file),
            device=self.device
        )
        print("[INFO] Re-ID Feature Extractor ready.")

    def extract_features(self, image_paths: list) -> np.ndarray:
        """
        Extracts L2-normalized embeddings for a list of image file paths.
        Returns np.ndarray of shape (N, 512).
        """
        if not image_paths:
            return np.array([])
        
        feats = self.extractor(image_paths)  # Tensor shape (N, 512)
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy()


def validate_market1501(extractor: ReIDExtractor, dataset_root: str = "datasets/raw/Market1501", num_identities: int = 20, seed: int = 42):
    """
    Validates identity feature separation on Market-1501 dataset.
    """
    print("\n" + "=" * 70)
    print("STEP 1: RE-ID PRETRAINED MODEL VALIDATION ON MARKET-1501 DATASET")
    print("=" * 70)

    candidate_dirs = [
        Path(dataset_root) / "Market-1501-v15.09.15-train" / "Market-1501-v15.09.15" / "bounding_box_train",
        Path(dataset_root) / "Market-1501-v15.09.15-test" / "Market-1501-v15.09.15" / "bounding_box_test",
        Path("datasets/raw/Market1501/Market-1501-v15.09.15-train/Market-1501-v15.09.15/bounding_box_train"),
        Path("3 Datasets/Market1501/Market-1501-v15.09.15-train/Market-1501-v15.09.15/bounding_box_train"),
    ]

    train_dir = None
    for cand in candidate_dirs:
        if cand.exists():
            train_dir = cand
            break

    if train_dir is None:
        raise FileNotFoundError(f"[ERROR] Could not find Market1501 bounding_box_train directory under '{dataset_root}'")

    files = [f for f in os.listdir(train_dir) if f.endswith(".jpg") and not f.startswith("-1") and not f.startswith("0000")]

    # Group files by Person ID (PID)
    id_map = defaultdict(list)
    for f in files:
        pid = f.split("_")[0]
        id_map[pid].append(f)

    # Filter PIDs with images from at least 2 different cameras
    valid_ids = {}
    for pid, f_list in id_map.items():
        cams = set(f.split("_")[1] for f in f_list)
        if len(cams) >= 2:
            valid_ids[pid] = f_list

    print(f"[INFO] Found {len(valid_ids)} valid identities with >= 2 camera views in Market1501.")

    random.seed(seed)
    sampled_pids = sorted(random.sample(list(valid_ids.keys()), min(num_identities, len(valid_ids))))

    sampled_img_paths = []
    img_pids = []
    pid_img_indices = defaultdict(list)

    for pid in sampled_pids:
        # Take up to 4 images per identity
        selected_files = valid_ids[pid][:4]
        for f in selected_files:
            idx = len(sampled_img_paths)
            sampled_img_paths.append(str(train_dir / f))
            img_pids.append(pid)
            pid_img_indices[pid].append(idx)

    print(f"[INFO] Sampled {len(sampled_pids)} identities ({len(sampled_img_paths)} total images). Extracting embeddings...")

    feats = extractor.extract_features(sampled_img_paths)  # Shape (N, 512)

    # Compute Cosine Similarity Matrix (N x N)
    sim_matrix = np.dot(feats, feats.T)

    intra_sims = []
    inter_sims = []

    N = len(sampled_img_paths)
    for i in range(N):
        for j in range(i + 1, N):
            sim = float(sim_matrix[i, j])
            if img_pids[i] == img_pids[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    intra_mean, intra_min, intra_max = np.mean(intra_sims), np.min(intra_sims), np.max(intra_sims)
    inter_mean, inter_min, inter_max = np.mean(inter_sims), np.min(inter_sims), np.max(inter_sims)
    margin = intra_mean - inter_mean

    print("\n--- MARKET-1501 COSINE SIMILARITY DISTRIBUTION ---")
    print(f"Intra-Identity (Same Person, Diff Cams) : Mean = {intra_mean:.4f} | Min = {intra_min:.4f} | Max = {intra_max:.4f}")
    print(f"Inter-Identity (Diff Persons)            : Mean = {inter_mean:.4f} | Min = {inter_min:.4f} | Max = {inter_max:.4f}")
    print(f"Separation Margin (Intra Mean - Inter)  : +{margin:.4f} ({margin*100:.2f}%)")

    # Display 5x5 Identity Average Cosine Similarity Matrix for first 5 identities
    first_5_pids = sampled_pids[:5]
    mean_pid_feats = []
    for pid in first_5_pids:
        indices = pid_img_indices[pid]
        pid_feats = feats[indices]
        mean_feat = np.mean(pid_feats, axis=0)
        mean_feat = mean_feat / np.linalg.norm(mean_feat)
        mean_pid_feats.append(mean_feat)

    mean_pid_feats = np.array(mean_pid_feats)  # (5, 512)
    matrix_5x5 = np.dot(mean_pid_feats, mean_pid_feats.T)

    print("\n--- 5x5 IDENTITY AVERAGE COSINE SIMILARITY MATRIX (First 5 Market1501 PIDs) ---")
    header = f"{'PID':<10}" + "".join([f"{pid:>10}" for pid in first_5_pids])
    print(header)
    print("-" * len(header))
    for i, p_row in enumerate(first_5_pids):
        row_str = f"{p_row:<10}"
        for j, p_col in enumerate(first_5_pids):
            val = matrix_5x5[i, j]
            row_str += f"{val:10.4f}"
        print(row_str)

    # Evaluation verdict
    if margin < 0.15:
        print("\n[WARNING] Model feature separation margin is low (< 0.15). Consider selecting another checkpoint.")
        return False
    else:
        print("\n[VERDICT] Model shows strong identity feature separation on Market1501! Proceeding to video tracks...")
        return True


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Re-ID embeddings for track crops")
    parser.add_argument("--crops-dir", type=str, default=None,
                        help="Path to folder containing crop subdirectories (e.g. reports/tracking/crops/real_pedestrians)")
    parser.add_argument("--json-path", type=str, default=None,
                        help="Path to attributes.json file to update with embeddings")
    parser.add_argument("--video-name", type=str, default=None,
                        help="Tên video (vd: 'real_pedestrians', 'store-aisle-detection')")
    parser.add_argument("--skip-market-validation", action="store_true",
                        help="Bỏ qua bước validate Market-1501 để chạy nhanh hơn")
    return parser.parse_args()


def process_video_tracks(extractor: ReIDExtractor, crops_dir: str, json_path: str):
    """
    Extracts mean-pooled L2-normalized embeddings for each track in crops_dir,
    updates attributes.json, and prints track similarity matrix.
    """
    print("\n" + "=" * 70)
    print("STEP 2: TRACK EMBEDDING AGGREGATION & MATRIX ANALYSIS")
    print("=" * 70)

    crops_path = Path(crops_dir)
    json_file = Path(json_path)

    if not crops_path.exists():
        raise FileNotFoundError(f"[ERROR] Crops directory not found: {crops_path}")

    json_data = {}
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load existing JSON '{json_file}': {e}")
            json_data = {}

    track_folders = sorted(
        [d for d in crops_path.iterdir() if d.is_dir() and d.name.startswith("track_")],
        key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else x.name
    )

    track_ids = []
    track_embeddings = []

    for folder in track_folders:
        t_id = int(folder.name.split("_")[1])
        crop_files = sorted([str(f) for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])

        if not crop_files:
            continue

        crop_feats = extractor.extract_features(crop_files)
        mean_emb = np.mean(crop_feats, axis=0)  # Mean pooling across crops
        mean_emb = mean_emb / np.linalg.norm(mean_emb)  # L2 normalize

        track_ids.append(t_id)
        track_embeddings.append(mean_emb)
        print(f"[PROC] Track {t_id:2d}: {len(crop_files):2d} crops -> 512-dim embedding computed.")

        str_id = str(t_id)
        emb_list = [round(float(v), 6) for v in mean_emb]
        if str_id in json_data:
            json_data[str_id]["embedding"] = emb_list
        else:
            json_data[str_id] = {"track_id": t_id, "embedding": emb_list}

    if json_file.parent.exists():
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n[OUTPUT] Updated '{json_file.resolve()}' with 'embedding' field ({len(track_ids)} tracks).")

        # Also sync to track_attributes.json or attributes.json if both exist in folder
        alt_json = json_file.parent / ("track_attributes.json" if json_file.name == "attributes.json" else "attributes.json")
        with open(alt_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"[OUTPUT] Synced embeddings to '{alt_json.resolve()}'.")

    # --- Compute Cosine Similarity Matrix ---
    if len(track_embeddings) > 0:
        sim_matrix = np.dot(track_embeddings, track_embeddings) if len(track_embeddings) == 1 else np.dot(track_embeddings, np.array(track_embeddings).T)

        print(f"\n--- TRACK COSINE SIMILARITY MATRIX ({len(track_ids)} tracks) ---")
        header_str = f"{'Track ID':<10}" + "".join([f"Track_{t_id:<6}" for t_id in track_ids])
        print(header_str)
        print("-" * len(header_str))

        high_sim_pairs = []

        for i, t1 in enumerate(track_ids):
            row_str = f"Track_{t1:<5}"
            for j, t2 in enumerate(track_ids):
                val = sim_matrix[i, j] if sim_matrix.ndim > 1 else sim_matrix[0]
                row_str += f"{val:12.4f}"
                if i < j and val > 0.7:
                    high_sim_pairs.append((t1, t2, val))
            print(row_str)

        print("\n--- ANALYSIS OF TRACK COSINE SIMILARITY MATRIX ---")
        if high_sim_pairs:
            print("[NOTICE] Track pairs with high cosine similarity (> 0.7):")
            for t1, t2, val in high_sim_pairs:
                print(f"  - Track {t1} vs Track {t2}: Cosine Sim = {val:.4f}")
        else:
            print("[OK] No track pairs exceed 0.7 cosine similarity threshold. All tracks show clear feature distinction.")


def main():
    args = parse_args()
    print("=" * 70)
    print("LEVEL 3: RE-IDENTIFICATION (RE-ID) FEATURE EMBEDDING PIPELINE")
    print("=" * 70)

    extractor = ReIDExtractor(model_name="osnet_x1_0", checkpoint_path=DEFAULT_CHECKPOINT_PATH)

    # Step 1: Validate on Market1501 (unless skipped)
    if not args.skip_market_validation:
        try:
            valid_model = validate_market1501(extractor, dataset_root="3 Datasets/Market1501", num_identities=20)
            if not valid_model:
                print("[STOP] Model validation failed.")
                sys.exit(1)
        except Exception as e:
            print(f"[NOTICE] Skipping Market1501 validation: {e}")

    # Resolve crops_dir and json_path
    if args.video_name:
        video_name = args.video_name
        crops_dir = f"reports/tracking/crops/{video_name}"
        json_path = f"reports/tracking/{video_name}/attributes.json"
    else:
        crops_dir = args.crops_dir or "reports/tracking/crops/real_pedestrians"
        json_path = args.json_path or "reports/tracking/real_pedestrians/attributes.json"

    # Step 2: Apply to video tracks
    process_video_tracks(extractor, crops_dir=crops_dir, json_path=json_path)


if __name__ == "__main__":
    main()
