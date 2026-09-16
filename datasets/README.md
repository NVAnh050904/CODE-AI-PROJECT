# Hướng Dẫn Cấu Trúc Thư Mục Datasets (Raw vs Processed)

Hệ thống quản lý dữ liệu đối tượng người đi bộ UPAR được phân chia thành **2 cấp độ riêng biệt**:

---

## 1. Dữ liệu thô ban đầu (Raw Datasets) — `datasets/raw/`

Thư mục `datasets/raw/` chứa các tập dữ liệu người đi bộ gốc thu thập từ các nguồn độc lập trước khi qua bước chuẩn hóa nhãn:

- **`datasets/raw/Market1501/`**: Tập dữ liệu Market-1501 (ảnh người đi bộ từ 6 góc quay camera CCTV siêu thị).
- **`datasets/raw/PA-100K/`**: Tập dữ liệu PA-100K (100,000 ảnh outdoor/indoor giám sát với nhãn thuộc tính phong phú).
- **`datasets/raw/PETA/`**: Tập dữ liệu PETA (Pedestrian Attribute dataset tổng hợp 19.000 ảnh).

---

## 2. Dữ liệu gộp đã xử lý để Huấn luyện (Processed Training Datasets) — `datasets/processed/`

Thư mục `datasets/processed/` chứa nhãn và dữ liệu đã qua tiền xử lý, gộp nhãn đồng nhất (UPAR UNIFIED Schema) dùng trực tiếp cho quá trình **Huấn luyện (Train)**, **Kiểm định (Val)** và **Đánh giá (Test)**:

- **`datasets/processed/UPAR_UNIFIED/annotations/`**:
  - `train.pkl`: Tập dữ liệu huấn luyện gộp (Train split - 101,959 mẫu).
  - `val.pkl`: Tập dữ liệu kiểm định (Validation split - 14,566 mẫu).
  - `test.pkl`: Tập dữ liệu kiểm thử (Test split - 29,131 mẫu).
  - `unified_annotations.pkl`: Tập tổng hợp toàn bộ 145,656 mẫu người đi bộ.
- **`datasets/processed/UPAR_UNIFIED/mapping/`**: Các file định nghĩa quy tắc mapping nhãn thuộc tính từ Market1501, PA-100K, PETA về chuẩn UPAR 40 thuộc tính.
- **`datasets/processed/UPAR_UNIFIED/scripts/`**: Các kịch bản chạy hợp nhất nhãn.

---

## 3. Mã nguồn nạp dữ liệu (PyTorch Dataset Loader) — `datasets/upar/`

- **[datasets/upar/loader.py](file:///c:/Users/ADMIN/OneDrive/Documents/GitHub/AI-Project/datasets/upar/loader.py)**: Module PyTorch `UPARDataset` tự động quét dữ liệu từ `datasets/raw/` và `datasets/processed/UPAR_UNIFIED/`, hỗ trợ nạp đa tiến trình (multi-threaded image loader) với hiệu năng cao.

---

> **Lưu ý tương thích**: Để đảm bảo các script cũ không bị ảnh hưởng, hai đường dẫn viết tắt `3 Datasets` và `UPAR_UNIFIED` tại gốc thư mục dự án được cấu hình dạng Directory Junction tự động trỏ tương ứng về `datasets/raw` và `datasets/processed/UPAR_UNIFIED`.
