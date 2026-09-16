import streamlit as st
import pandas as pd
import sqlite3
import subprocess
import sys
import os
import shutil
from pathlib import Path
import time

# Khởi tạo DB SQLite
def init_db():
    conn = sqlite3.connect('recognition_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, gender TEXT, clothing TEXT, accessory TEXT, confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

st.set_page_config(page_title="PAR", layout="wide")
st.title("🛡️ Hệ Thống Nhận Diện Thuộc Tính Người Đi Bộ")
st.markdown("---")

# Cấu trúc thư mục dự án của bạn
base_dir = Path(os.getcwd())
test_videos_dir = base_dir / "tracking" / "test_videos"
demo_output_dir = base_dir / "reports" / "tracking" / "demo"
test_videos_dir.mkdir(parents=True, exist_ok=True)
demo_output_dir.mkdir(parents=True, exist_ok=True)

# Khởi tạo session state để lưu trữ video đang phát giữa các lần rerun (như khi bấm nút Lọc)
if "active_video_name" not in st.session_state:
    st.session_state["active_video_name"] = None
if "active_video_path" not in st.session_state:
    st.session_state["active_video_path"] = None

# SIDEBAR: Cấu hình
st.sidebar.header("⚙️ Cấu Hình Hệ Thống")
source_type = st.sidebar.selectbox("Chọn nguồn", ["Tải lên file video"])
conf_threshold = st.sidebar.slider("Ngưỡng nhận diện (YOLO Conf)", 0.1, 1.0, 0.35, 0.05)
crop_step = st.sidebar.slider("Tần suất lấy crop (Every N frames)", 1, 30, 5, 1)

uploaded_file = st.sidebar.file_uploader("Chọn file video (.mp4)", type=["mp4"])
run_system = st.sidebar.button("🚀 Khởi chạy Pipeline AI", type="primary")

# CHÍNH: Xử lý video
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Xử Lý & Kết Quả")
    status_msg = st.empty()
    
    if run_system:
        if uploaded_file is not None:
            # 1. Lưu file video vào thư mục test_videos của dự án
            video_name = os.path.splitext(uploaded_file.name)[0]
            video_path = test_videos_dir / uploaded_file.name
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            status_msg.info(f"Đang chạy toàn bộ Pipeline 5 bước cho video: {video_name}... (Có thể mất vài phút)")
            
            # 2. Gọi script pipeline của nhóm bằng subprocess (dùng sys.executable để đảm bảo dùng đúng venv)
            pipeline_cmd = [
                sys.executable, str(base_dir / "tracking" / "run_pipeline.py"),
                "--video-name", video_name,
                "--conf", str(conf_threshold),
                "--every-n-frames", str(crop_step)
            ]
            
            with st.spinner('AI đang xử lý Tracking, Attribute & Re-ID...'):
                start_time = time.time()
                
                # Ép môi trường chạy ngầm sử dụng mã hóa UTF-8 để chống lỗi đường dẫn tiếng Việt
                my_env = os.environ.copy()
                my_env["PYTHONIOENCODING"] = "utf-8"
                
                # Chạy pipeline thực tế
                process = subprocess.run(
                    pipeline_cmd, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8',
                    env=my_env
                )
                elapsed = time.time() - start_time
                
            if process.returncode == 0:
                # 3. Tìm và hiển thị video kết quả
                demo_file_name = f"demo_{video_name}_v2.mp4"
                demo_file_path = demo_output_dir / demo_file_name
                
                if demo_file_path.exists():
                    converted_path = demo_output_dir / f"web_ready_{demo_file_name}"
                    
                    if not converted_path.exists() or converted_path.stat().st_size == 0:
                        status_msg.info("⏳ AI đã xong! Đang chuyển đổi mã hóa video sang chuẩn H.264 Web-Ready...")
                        
                        # Thử dùng imageio_ffmpeg trực tiếp (Nhanh & 100% không lỗi)
                        try:
                            import imageio_ffmpeg
                            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                            ff_cmd = [
                                ffmpeg_exe, "-y", "-i", str(demo_file_path),
                                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-an", str(converted_path)
                            ]
                            subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        except Exception as e_ff:
                            # Thử dùng moviepy nếu ffmpeg gặp sự cố
                            try:
                                try:
                                    from moviepy import VideoFileClip
                                except ImportError:
                                    from moviepy.editor import VideoFileClip
                                clip = VideoFileClip(str(demo_file_path))
                                clip.write_videofile(str(converted_path), codec="libx264", audio=False, logger=None)
                                clip.close()
                            except Exception as e_mv:
                                status_msg.warning(f"Đang phát video bản gốc... (Lỗi: {e_mv})")

                    final_video_path = converted_path if (converted_path.exists() and converted_path.stat().st_size > 0) else demo_file_path
                    status_msg.success(f"✅ Hoàn tất toàn bộ trong {elapsed:.1f} giây! Mời bạn xem kết quả bên dưới.")
                    
                    # Lưu thông tin video vừa xử lý vào session state để không bị biến mất khi bấm Lọc
                    st.session_state["active_video_name"] = video_name
                    st.session_state["active_video_path"] = str(final_video_path)
                else:
                    st.error("Không tìm thấy file video kết quả đầu ra trong thư mục reports/tracking/demo/.")
            else:
                status_msg.error("❌ Có lỗi xảy ra trong quá trình chạy mô hình AI:")
                st.code(process.stderr if process.stderr else process.stdout)
        else:
            status_msg.warning("⚠️ Vui lòng tải lên một file video!")

    # HIỂN THỊ VIDEO & BẢNG THỐNG KÊ (Duy trì liên tục qua st.session_state kể cả khi bấm Lọc)
    active_path = st.session_state.get("active_video_path")
    active_name = st.session_state.get("active_video_name")

    if active_path and Path(active_path).exists():
        st.markdown(f"#### 📺 Video Kết Quả Pipeline (`{active_name}`)")
        st.video(active_path)

        summary_csv = base_dir / "reports" / "tracking" / active_name / "tracked_persons_summary.csv" if active_name else None
        if summary_csv and summary_csv.exists():
            st.markdown("#### 📊 Bảng Thống Kê Thuộc Tính Các Đối Tượng Trong Video")
            df_summary = pd.read_csv(summary_csv)
            st.dataframe(df_summary[["track_id", "gender", "age", "upper_color", "lower_color", "bag"]], use_container_width=True)
    elif not run_system:
        st.info("Bấm 'Khởi chạy Pipeline AI' để hệ thống bắt đầu chạy 5 bước xử lý.")

with col2:
    st.subheader("📋 Trạng Thái Hệ Thống")
    st.metric(label="Ngưỡng YOLO hiện tại", value=f"{conf_threshold}")
    st.markdown("---")
    st.write("**Các bước AI sẽ chạy:**")
    st.write("1. Tracking (YOLOv8 + ByteTrack)")
    st.write("2. Trích xuất Crop người")
    st.write("3. Nhận diện 40 thuộc tính (UPAR)")
    st.write("4. Trích xuất đặc trưng Re-ID")
    st.write("5. Render video kết quả")

st.markdown("---")

# Mở rộng: Tính năng Lọc & Truy vấn Đối tượng (Level 5 Person Retrieval)
with st.expander("🔍 Engine Lọc & Truy Vấn Đối Tượng Trong CSDL Video (Person Retrieval Level 5)"):
    st.markdown("Tìm kiếm đối tượng người đi bộ theo thuộc tính trong CSDL `person_database.json`:")
    
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        q_gender = st.selectbox("Giới tính:", ["Tất cả", "Female", "Male"])
    with col_q2:
        q_upper = st.selectbox("Màu áo:", ["Tất cả", "Black", "Blue", "Brown", "Green", "Grey", "Red", "White", "Yellow"])
    with col_q3:
        q_lower = st.selectbox("Màu quần/váy:", ["Tất cả", "Black", "Blue", "Brown", "Green", "Grey", "Red", "White", "Yellow"])
        
    if st.button("🔎 Thực Hiện Lọc"):
        cmd_q = [sys.executable, str(base_dir / "tracking" / "query_persons.py")]
        if q_gender != "Tất cả":
            cmd_q.extend(["--gender", q_gender])
        if q_upper != "Tất cả":
            cmd_q.extend(["--upper_color", q_upper])
        if q_lower != "Tất cả":
            cmd_q.extend(["--lower_color", q_lower])
            
        with st.spinner("Đang truy vấn CSDL..."):
            res_q = subprocess.run(cmd_q, capture_output=True, text=True, encoding="utf-8")
            
        if res_q.returncode == 0:
            st.success("Truy vấn thành công!")
            st.code(res_q.stdout, language="text")
            
            # Hiển thị chính xác ảnh grid kết quả vừa được tạo nếu có
            grid_dir = base_dir / "reports" / "tracking" / "query_results"
            
            target_img_path = None
            # Trích xuất đường dẫn file trực tiếp từ stdout của query_persons.py cho lần chạy hiện tại
            for line in res_q.stdout.splitlines():
                if "[OUTPUT GRID IMAGE]" in line and "'" in line:
                    parts = line.split("'")
                    if len(parts) >= 2:
                        cand = base_dir / parts[1]
                        if cand.exists():
                            target_img_path = cand
                            break

            if target_img_path and target_img_path.exists():
                st.image(str(target_img_path), caption=f"Lưới ảnh kết quả lọc đối tượng ({target_img_path.name})", use_container_width=True)
            else:
                st.warning("⚠️ Không tìm thấy đối tượng nào thỏa mãn bộ lọc (không sinh ảnh kết quả).")
        else:
            st.error("Có lỗi xảy ra khi truy vấn CSDL.")
