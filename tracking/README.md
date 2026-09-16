# Module Video Tracking, Attribute Aggregation, Re-ID & Hybrid Matching

Module này mở rộng hệ thống nhận diện thuộc tính người đi bộ UPAR từ ảnh tĩnh thành **Pipeline Phân tích Video Người đi bộ Toàn diện (4 Level)**, kết hợp giữa Tracking ngắn hạn, Gom nhóm Thuộc tính, Re-Identification (OSNet) và Hybrid Matching.

> **Báo cáo Kỹ thuật & Thực nghiệm Chuyên sâu**: Xem chi tiết phương pháp luận, quá trình sửa lỗi pseudo-replication và kết quả đánh giá LOOCV tại [`tracking/TECHNICAL_REPORT.md`](file:///c:/Users/ADMIN/OneDrive/Documents/GitHub/AI-Project/tracking/TECHNICAL_REPORT.md).

---

## 1. Cài đặt Thư viện Dependencies

```powershell
pip install ultralytics opencv-python torchreid networkx pandas
```

* `ultralytics`: Tích hợp sẵn YOLOv8 và ByteTrack/BoT-SORT (`bytetrack.yaml`).
* `torchreid`: Trích xuất đặc trưng Re-ID (OSNet `osnet_x1_0`).
* `networkx`: Xây dựng đồ thị liên thông gom nhóm Identity đối tượng.

---

## 2. Hướng dẫn Vận hành theo 4 Level & Pipeline Tự Động

### Script Điều Phối Tự Động 1 Lệnh (`run_pipeline.py`)

Để chạy toàn bộ Pipeline từ Video gốc đến Video Demo V2 hoàn chỉnh (bao gồm 6 bước tự động: Tracking, Crop Extraction, Attribute Aggregation, Re-ID Embedding, Combined Demo Video Generation và Tự động nạp CSDL `person_database.json`) chỉ với **1 câu lệnh duy nhất**:

```powershell
python tracking/run_pipeline.py --video-name store-aisle-detection
```

Các tham số tùy chọn:
* `--video-name`: Tên video trong `tracking/test_videos/` (ví dụ: `store-aisle-detection`, `real_pedestrians`, `classroom`, `worker-zone-detection`, `people-detection`, `face-demographics-walking-and-pause`, `vtest`).
* `--video-path`: (Tùy chọn) Đường dẫn tùy chỉnh tới file video nếu nằm ngoài `tracking/test_videos/`.
* `--conf`: Tương quan confidence threshold cho YOLOv8 (Mặc định: `0.35`).
* `--every-n-frames`: Bước nhảy trích xuất crop ảnh người (Mặc định: `5`).

> **Tự động nhận diện môi trường ảo**: Script tự động ưu tiên kích hoạt `.venv\Scripts\python.exe` nếu tồn tại để đảm bảo nạp đúng mô hình PyTorch & CUDA.
>
> **Lưu ý về File Video Trùng lặp**: `people-detection.mp4` và `real_pedestrians.mp4` là cùng 1 nội dung video (phát hiện trùng lặp ngày 14/09/2026 qua mã checksum MD5 `69dafa7fd143c2bee7f216431304b071`), giữ cả 2 file để không phá vỡ các tham chiếu cũ, nhưng **KHÔNG đưa `people-detection` vào `OFFICIAL_VIDEOS`** để tránh đếm trùng.

---

### Level 1 — Detection + Tracking (`track.py`)
Phát hiện người đi bộ và duy trì `track_id` ngắn hạn trên video stream:

```powershell
# Tracking từ file video:
python tracking/track.py --source tracking/test_videos/real_pedestrians.mp4 --save-video

# Tracking từ webcam:
python tracking/track.py --source 0 --show
```
* **Output**: `reports/tracking/<ten_video>/tracks.csv` và `reports/tracking/<ten_video>/tracked.mp4`.
* **Model weights**: Tự động lưu/tải tại `checkpoints/yolov8n.pt`.

---

### Level 2 — Trích Crop & Gom Thuộc tính (`extract_crops.py` & `track_attributes.py`)

#### Bước 2.1 — Trích Crop ảnh người đi bộ theo Track ID:
```powershell
python tracking/extract_crops.py \
  --video tracking/test_videos/real_pedestrians.mp4 \
  --csv reports/tracking/real_pedestrians/tracks.csv \
  --output-dir reports/tracking/crops/real_pedestrians \
  --every-n-frames 5 \
  --clean
```
* **Output**: Ảnh crop người theo cấu trúc `reports/tracking/crops/<ten_video>/track_<id>/frame_<n>.jpg`.

#### Bước 2.2 — Gán & Gom nhóm Thuộc tính UPAR qua thời gian:
```powershell
python tracking/track_attributes.py \
  --crops-dir reports/tracking/crops/real_pedestrians \
  --tracks-csv reports/tracking/real_pedestrians/tracks.csv \
  --checkpoint checkpoints/hydraplus_upar_best.pth \
  --output-dir reports/tracking/real_pedestrians \
  --min-frames 3
```
* **Output**: 
  - `reports/tracking/<ten_video>/attributes.csv`: Bảng thuộc tính Top-1 cho mỗi `track_id`.
  - `reports/tracking/<ten_video>/attributes.json`: Chi tiết multi-label active và 40 xác suất raw.
  - `reports/tracking/tracked_persons_summary.csv`: Bảng tổng hợp đối tượng (Track metadata + Attributes).

---

### Level 3 — Re-ID Feature Embedding Extractor & Validation

#### Bước 3.1 — Trích 512-dim Embedding & Validate Market1501 Benchmark (`reid_embedding.py`):
```powershell
python tracking/reid_embedding.py
```
* Đánh giá phân phối Cosine Similarity Intra-ID vs Inter-ID trên dataset Market1501 (Separation Margin: **+34.72%**).
* Tự động tính mean-pooled embedding 512 chiều cho từng `track_id` và cập nhật vào `reports/tracking/<ten_video>/attributes.json`.

#### Bước 3.2 — Đánh giá Domain Gap trên Video Thực (`reid_validate_domain.py`):
```powershell
python tracking/reid_validate_domain.py
```
* Đánh giá phân phối Cosine Similarity ở cấp độ frame-level trên video thực `real_pedestrians.mp4`.

#### Bước 3.3 — Đánh giá Re-entry & Chống Pseudo-Replication (`reid_validate_reentry.py`):
```powershell
python tracking/reid_validate_reentry.py \
  --crops-dir reports/tracking/crops/store-aisle-detection \
  --gt-csv reports/tracking/store-aisle-detection/reentry_ground_truth.csv
```
* Đánh giá EER bài toán Re-entry (người bị occlusion/rời khung hình) kết hợp gom nhóm Identity Độc lập & Bootstrap 1,000 lần.

#### Bước 3.4 — Benchmark Multi-Video Domain (`reid_validate_reentry_combined.py`):
```powershell
python tracking/reid_validate_reentry_combined.py
```
* Đánh giá Re-ID trên 4 video benchmark ($N=12$ sự kiện re-entry độc lập), báo cáo song song **Micro EER (9.67%)** và **Macro EER (13.70%)**.

---

### Level 4 — Hybrid Matching & Cross-Validation (`hybrid_matching.py`)
Kết hợp Re-ID Similarity + Attribute Similarity + Time Penalty theo công thức Hybrid Score:

$$S_{\text{hybrid}} = w_1 S_{\text{reid}} + w_2 S_{\text{attr}} - w_3 P_{\text{time}}$$

```powershell
python tracking/hybrid_matching.py
```
* Sử dụng **Leave-One-Out Cross-Validation (LOOCV)** ở cấp độ video để tìm trọng số tối ưu tránh Data Leakage.
* **Kết quả**: Re-ID 512 chiều đóng vai trò cốt lõi ($w_1=0.80$). Thuộc tính UPAR không làm tăng Re-entry EER do nhiễu nền CCTV, nhưng dùng rất tốt cho bài toán **Person Retrieval / Attribute Querying**.

---

### Demo Combined Video Generation (`demo_combined.py`)
Tạo video demo hợp nhất kết hợp trực quan Level 1 (Tracking) + Level 2 (Attribute - Bảng thuộc tính UPAR 11-Heads) + Level 3 (Re-ID):

```powershell
# Chạy demo cho video mặc định (store-aisle-detection):
python tracking/demo_combined.py --video-name store-aisle-detection

# Chạy demo cho bất kỳ video nào khác (chế độ graceful tự động nếu không có reentry_ground_truth.csv):
python tracking/demo_combined.py --video-name real_pedestrians
python tracking/demo_combined.py --video-name classroom
python tracking/demo_combined.py --video-name worker-zone-detection
python tracking/demo_combined.py --video-name people-detection
```

* **Điểm nổi bật V2**:
  - **Canvas 960x360 (Không che khuất)**: Mở rộng 320px lề phải làm Side Info Panel hiển thị ĐẦY ĐỦ 11 head thuộc tính UPAR (Age, Gender, Hair, Glasses, Hat, Upper, Lower, Bag) cho identity đang chọn.
  - **Quy tắc chọn Identity thông minh**: Tự động hiển thị đối tượng có Bbox lớn nhất trong frame; tự động chuyển sang hiển thị identity vừa được nhận lại kèm thẻ `RE-ID EVENT` khi có sự kiện Re-ID xảy ra.
* **Output**:
  - `reports/tracking/demo/demo_combined_v2_full_attributes.mp4` (Video demo hợp nhất 960x360, 30 FPS, < 30 MB)
  - `reports/tracking/demo/demo_{video-name}_v2.mp4` (Video demo hợp nhất cho từng video test)
  - `reports/tracking/demo/v2_screenshot_*.jpg` (4 ảnh chụp màn hình minh chứng các khoảnh khắc Re-ID và theo dõi thuộc tính)

---

## 3. Cấu trúc File Module `tracking/`

```text
tracking/
├── __init__.py
├── README.md                               # Hướng dẫn sử dụng module tracking
├── TECHNICAL_REPORT.md                     # Báo cáo kỹ thuật & kết quả nghiên cứu 4 Level
├── run_pipeline.py                         # Orchestrator tự động hóa 6 bước (Tracking, Crop, Attribute, Embedding, Demo, CSDL Auto-Index)
├── track.py                                # Level 1: YOLOv8 + ByteTrack tracking pipeline
├── extract_crops.py                        # Level 2: Trích crop ảnh người theo track_id
├── track_attributes.py                     # Level 2: Gom nhóm xác suất 40 thuộc tính UPAR
├── reid_embedding.py                       # Level 3: Trích xuất OSNet 512-dim embedding
├── reid_validate_domain.py                 # Level 3: Validate similarity trên real video
├── reid_validate_reentry.py                # Level 3: Validate Re-entry & Chống Pseudo-replication
├── reid_validate_reentry_combined.py       # Level 3: Benchmark Re-ID trên 4 video
├── build_person_database.py                # Level 4: Gom nhóm identity Graph NetworkX & xây dựng person_database.json
├── query_persons.py                        # Level 5: Engine lọc đối tượng theo thuộc tính UPAR & Re-ID Target Image
├── hybrid_matching.py                      # Level 4: Hybrid Score & LOOCV Grid Search
├── demo_combined.py                        # Video Demo Hợp nhất V2 (Canvas 960x360 + UPAR Side Panel)
└── test_videos/                            # Thư mục lưu trữ video thử nghiệm mẫu (.mp4, .avi)
```
