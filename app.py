"""
app.py
======
Streamlit Frontend Application for UPAR Multi-Head Pedestrian Attribute Recognition
& Video Person Retrieval System.

Connects to Backend modules:
- tracking/run_pipeline.py (Level 1-4 Pipeline Orchestrator)
- tracking/query_persons.py (Level 5 Person Retrieval & Filtering Engine)
- tracking/build_person_database.py (Database Builder)
"""

import os
import sys
import time
import json
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Database Initialization (SQLite)
# -----------------------------------------------------------------------------
DB_PATH = "recognition_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action_type TEXT,
            video_name TEXT,
            gender TEXT,
            upper_color TEXT,
            lower_color TEXT,
            total_tracks INTEGER,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_activity(action_type: str, video_name: str = "N/A", gender: str = "All", 
                 upper_color: str = "All", lower_color: str = "All", 
                 total_tracks: int = 0, status: str = "Success"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO logs (timestamp, action_type, video_name, gender, upper_color, lower_color, total_tracks, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now_str, action_type, video_name, gender, upper_color, lower_color, total_tracks, status))
    conn.commit()
    conn.close()

def fetch_history_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df

init_db()

# -----------------------------------------------------------------------------
# 2. Streamlit UI Setup & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UPAR Surveillance & Person Retrieval System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Hệ Thống Nhận Diện Thuộc Tính Người Đi Bộ UPAR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Video Surveillance Tracking, Attribute Recognition & Person Retrieval System</div>', unsafe_allow_html=True)

base_dir = Path(os.getcwd())
test_videos_dir = base_dir / "tracking" / "test_videos"
demo_output_dir = base_dir / "reports" / "tracking" / "demo"
db_json_path = base_dir / "reports" / "tracking" / "person_database.json"

test_videos_dir.mkdir(parents=True, exist_ok=True)
demo_output_dir.mkdir(parents=True, exist_ok=True)

# Helper to locate python executable
def get_python_exec():
    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Global Configs
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Điều Khiển & Cấu Hình")

app_mode = st.sidebar.radio(
    "Chọn Chế Độ Chạy:",
    [
        "1. Pipeline Video Tracking & Attributes",
        "2. Lọc & Truy Vấn Đối Tượng (Person Retrieval)",
        "3. Lịch Sử Nhật Ký (System Logs)"
    ]
)

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# 4. Mode 1: Video Pipeline Execution
# -----------------------------------------------------------------------------
if app_mode == "1. Pipeline Video Tracking & Attributes":
    st.subheader("🎥 Xử Lý Video Surveillance (Level 1 - Level 4)")
    
    st.sidebar.subheader("Cấu Hình Pipeline AI")
    conf_threshold = st.sidebar.slider("Ngưỡng nhận diện (YOLO Conf)", 0.1, 1.0, 0.35, 0.05)
    crop_step = st.sidebar.slider("Tần suất trích crop (Every N frames)", 1, 30, 5, 1)
    
    source_option = st.sidebar.radio("Nguồn Video:", ["Tải file video lên", "Chọn video mẫu sẵn có"])
    
    selected_video_name = None
    uploaded_file = None
    
    if source_option == "Tải file video lên":
        uploaded_file = st.sidebar.file_uploader("Tải lên file video (.mp4, .avi)", type=["mp4", "avi"])
        if uploaded_file is not None:
            selected_video_name = os.path.splitext(uploaded_file.name)[0]
    else:
        existing_videos = [f.name for f in test_videos_dir.glob("*.mp4")] + [f.name for f in test_videos_dir.glob("*.avi")]
        if existing_videos:
            video_choice = st.sidebar.selectbox("Danh sách video mẫu:", existing_videos)
            selected_video_name = os.path.splitext(video_choice)[0]
        else:
            st.sidebar.info("Không tìm thấy video mẫu trong tracking/test_videos/")
            
    run_button = st.sidebar.button("🚀 Khởi chạy Pipeline AI", type="primary")

    col1, col2 = st.columns([2, 1])

    with col1:
        status_box = st.empty()
        
        if run_button:
            if selected_video_name is None:
                st.error("Vui lòng tải lên hoặc chọn 1 video để khởi chạy.")
            else:
                if uploaded_file is not None:
                    target_video_path = test_videos_dir / uploaded_file.name
                    with open(target_video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Đã lưu video tải lên: {uploaded_file.name}")

                status_box.info(f"Đang kích hoạt Pipeline AI cho video: `{selected_video_name}`...")

                cmd = [
                    get_python_exec(),
                    str(base_dir / "tracking" / "run_pipeline.py"),
                    "--video-name", selected_video_name,
                    "--conf", str(conf_threshold),
                    "--every-n-frames", str(crop_step)
                ]

                progress_bar = st.progress(0)
                start_t = time.time()
                
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                    
                    logs_output = []
                    while True:
                        line = proc.stdout.readline()
                        if not line and proc.poll() is not None:
                            break
                        if line:
                            logs_output.append(line)
                            if "[STEP 1" in line:
                                progress_bar.progress(20)
                            elif "[STEP 2" in line:
                                progress_bar.progress(40)
                            elif "[STEP 3" in line:
                                progress_bar.progress(60)
                            elif "[STEP 4" in line:
                                progress_bar.progress(80)
                            elif "[STEP 5" in line:
                                progress_bar.progress(95)

                    proc.wait()
                    elapsed = time.time() - start_t
                    progress_bar.progress(100)

                    if proc.returncode == 0:
                        status_box.success(f"Hoàn thành Pipeline trong {elapsed:.2f} giây!")
                        log_activity(action_type="Run Pipeline", video_name=selected_video_name, status="Success")
                    else:
                        status_box.error("Có lỗi xảy ra khi chạy Pipeline.")
                        st.text_area("Log Lỗi:", "".join(logs_output), height=200)
                        log_activity(action_type="Run Pipeline", video_name=selected_video_name, status="Failed")

                except Exception as e:
                    status_box.error(f"Lỗi khởi chạy lệnh: {e}")
                    log_activity(action_type="Run Pipeline", video_name=selected_video_name, status=f"Error: {e}")

        # Display Output Video if available
        demo_v2_path = demo_output_dir / f"demo_{selected_video_name}_v2.mp4"
        general_demo_path = demo_output_dir / "demo_combined_v2_full_attributes.mp4"

        video_to_show = None
        if selected_video_name and demo_v2_path.exists():
            video_to_show = demo_v2_path
        elif general_demo_path.exists():
            video_to_show = general_demo_path

        if video_to_show and video_to_show.exists():
            st.markdown("#### 📺 Video Trực Quan Hóa Kết Quả (Demo Output)")
            st.video(str(video_to_show))
        else:
            st.info("Chưa có video demo output. Hãy nhấn **Khởi chạy Pipeline AI** để tạo video.")

    with col2:
        st.markdown("#### 📊 Bảng Thống Kê Attributes per Track")
        
        summary_csv = base_dir / "reports" / "tracking" / (selected_video_name if selected_video_name else "") / "tracked_persons_summary.csv"
        alt_csv = base_dir / "reports" / "tracking" / "tracked_persons_summary.csv"
        
        target_csv = summary_csv if summary_csv.exists() else (alt_csv if alt_csv.exists() else None)

        if target_csv and target_csv.exists():
            df_tracks = pd.read_csv(target_csv)
            st.metric("Tổng Số Tracks Khởi Tạo", len(df_tracks))
            st.dataframe(df_tracks[["track_id", "gender", "age", "upper_color", "lower_color", "bag"]], height=350, use_container_width=True)
            
            csv_bytes = df_tracks.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải Bảng Thuộc Tính (.CSV)",
                data=csv_bytes,
                file_name=f"{selected_video_name}_attributes.csv",
                mime="text/csv"
            )
        else:
            st.caption("Chưa có bảng dữ liệu attributes cho video này.")

# -----------------------------------------------------------------------------
# 5. Mode 2: Person Retrieval & Filtering (Level 5)
# -----------------------------------------------------------------------------
elif app_mode == "2. Lọc & Truy Vấn Đối Tượng (Person Retrieval)":
    st.subheader("🔍 Engine Lọc & Truy Vấn Đối Tượng Video (Level 5 Person Retrieval)")
    st.markdown("Tìm kiếm người đi bộ theo **nhãn thuộc tính tùy chọn** hoặc **so khớp ảnh mẫu Query Target** qua cơ sở dữ liệu `person_database.json`.")

    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
        filter_gender = st.selectbox("Giới Tính (Gender):", ["Tất cả", "Female", "Male"])
        filter_upper_color = st.selectbox("Màu Áo (Upper Color):", ["Tất cả", "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow"])

    with col_filter2:
        filter_lower_color = st.selectbox("Màu Quần/Váy (Lower Color):", ["Tất cả", "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow"])
        filter_bag = st.selectbox("Túi Xách / Balo (Bag):", ["Tất cả", "Backpack", "Bag", "None"])

    with col_filter3:
        filter_glasses = st.selectbox("Kính Mắt (Glasses):", ["Tất cả", "Normal Glasses", "Sunglasses", "None"])
        filter_hat = st.selectbox("Mũ (Hat):", ["Tất cả", "Hat", "No Hat"])

    st.markdown("---")
    st.markdown("##### 📷 Truv Vấn Theo Ảnh Mẫu Target (Re-ID Query Image - Optional)")
    query_image_file = st.file_uploader("Tải lên ảnh mẫu người cần tìm:", type=["jpg", "png", "jpeg"])

    btn_search = st.button("🔎 Thực Hiện Lọc & Truy Vấn", type="primary")

    if btn_search:
        if not db_json_path.exists():
            st.warning("Chưa có CSDL `person_database.json`. Đang tiến hành tự động khởi tạo...")
            build_cmd = [get_python_exec(), str(base_dir / "tracking" / "build_person_database.py"), "--rebuild-all"]
            subprocess.run(build_cmd, check=True)

        cmd_query = [get_python_exec(), str(base_dir / "tracking" / "query_persons.py")]
        
        if filter_gender != "Tất cả":
            cmd_query.extend(["--gender", filter_gender])
        if filter_upper_color != "Tất cả":
            cmd_query.extend(["--upper_color", filter_upper_color])
        if filter_lower_color != "Tất cả":
            cmd_query.extend(["--lower_color", filter_lower_color])
        if filter_bag != "Tất cả":
            cmd_query.extend(["--bag", filter_bag])
        if filter_glasses != "Tất cả":
            cmd_query.extend(["--glasses", filter_glasses])
        if filter_hat != "Tất cả":
            cmd_query.extend(["--hat", filter_hat])

        temp_query_img_path = None
        if query_image_file is not None:
            temp_dir = base_dir / "reports" / "tracking" / "query_results"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_query_img_path = temp_dir / "temp_query_input.jpg"
            with open(temp_query_img_path, "wb") as f:
                f.write(query_image_file.getbuffer())
            cmd_query.extend(["--query-image", str(temp_query_img_path)])

        with st.spinner("Đang tìm kiếm và xếp hạng đối tượng trùng khớp trong CSDL..."):
            res = subprocess.run(cmd_query, capture_output=True, text=True, encoding="utf-8")

        if res.returncode == 0:
            st.success("Truy vấn thành công!")
            st.code(res.stdout, language="text")

            log_activity(
                action_type="Query Persons",
                gender=filter_gender,
                upper_color=filter_upper_color,
                lower_color=filter_lower_color,
                status="Success"
            )

            # Display generated Image Grid if exists
            query_results_dir = base_dir / "reports" / "tracking" / "query_results"
            grid_images = sorted(list(query_results_dir.glob("query_result_*.png")), key=os.path.getmtime, reverse=True)
            
            if grid_images:
                st.markdown("#### 🖼️ Lưới Ảnh Đối Tượng Khớp Truy Vấn")
                latest_grid = grid_images[0]
                st.image(str(latest_grid), caption=f"Kết quả truy vấn: {latest_grid.name}", use_container_width=True)
        else:
            st.error("Lỗi thực thi truy vấn.")
            st.code(res.stderr, language="text")

# -----------------------------------------------------------------------------
# 6. Mode 3: System Logs & SQLite History
# -----------------------------------------------------------------------------
elif app_mode == "3. Lịch Sử Nhật Ký (System Logs)":
    st.subheader("📜 Nhật Ký Hoạt Động & Lịch Sử Truy Vấn (SQLite DB)")
    
    df_logs = fetch_history_logs()
    
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True, height=450)
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            st.metric("Tổng Số Lần Vận Hành System", len(df_logs))
        with col_db2:
            success_count = len(df_logs[df_logs["status"] == "Success"])
            st.metric("Tỷ Lệ Chạy Thành Công", f"{success_count / len(df_logs) * 100:.1f}%")
    else:
        st.info("Chưa có nhật ký hoạt động nào được ghi nhận.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("UPAR Multi-Head Pedestrian Attribute Recognition & Video Person Retrieval System | Powered by PyTorch, YOLOv8 & Streamlit")
