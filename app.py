import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="在线智能交通监控系统", layout="wide")


# 加载模型
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')


model = load_model()

st.title("🚗 在线多模态交通识别系统")
st.markdown("---")

# 1. 上传模块
uploaded_file = st.sidebar.file_uploader("点击上传视频文件", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # 创建临时文件保存上传的内容
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📺 实时检测画面")
        st_frame = st.empty()  # 动态更新视频帧

    with col2:
        st.subheader("📊 实时检测清单")
        st_data = st.empty()

    # 2. 推理模块
    cap = cv2.VideoCapture(tfile.name)
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 画面缩放
        frame = cv2.resize(frame, (640, 360))
        results = model.predict(frame, conf=0.25, verbose=False)

        # 记录数据
        current_cars = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in ['car', 'bus', 'truck']:
                current_cars += 1

        all_detections.append(current_cars)

        # 更新前端展示
        annotated_frame = results[0].plot()
        st_frame.image(annotated_frame, channels="BGR")

        # 显示简单的实时统计
        st_data.write(f"当前画面内车辆数: **{current_cars}**")

    cap.release()
    st.success("视频处理完成！")
else:
    st.info("请在左侧上传一个视频文件开始识别。")