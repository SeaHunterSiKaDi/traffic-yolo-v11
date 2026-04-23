import streamlit as st
from ultralytics import YOLO
import pandas as pd
import tempfile
import PIL.Image  # 使用 Python 自带的图像库代替 cv2
import os

st.set_page_config(page_title="在线智能交通监控系统", layout="wide")

@st.cache_resource
def load_model():
    # 强制模型只使用 CPU，减少对底层驱动的依赖
    return YOLO('yolo11n.pt')

model = load_model()

st.title("🚗 在线多模态交通识别系统 (云端稳定版)")

uploaded_file = st.sidebar.file_uploader("点击上传视频文件", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st_frame = st.empty()
    st_data = st.empty()
    
    # 关键修改：使用 model.predict 的 stream 模式，避免手动调用 cv2.VideoCapture
    # 这会直接通过 ultralytics 内部逻辑处理视频流
    results = model.predict(source=tfile.name, conf=0.25, stream=True)
    
    for result in results:
        # 获取带有检测框的图像 (numpy 数组)
        res_plotted = result.plot()
        
        # 将 BGR 转换为 RGB（PIL 库不需要底层的 libGL）
        img_rgb = res_plotted[:, :, ::-1]
        img_pil = PIL.Image.fromarray(img_rgb)
        
        # 在页面显示
        st_frame.image(img_pil, caption="实时检测中...", use_column_width=True)
        
        # 统计当前帧车辆
        current_cars = 0
        for box in result.boxes:
            if model.names[int(box.cls[0])] in ['car', 'bus', 'truck']:
                current_cars += 1
        
        st_data.write(f"当前画面内车辆数: **{current_cars}**")

    st.success("视频处理完成！")
else:
    st.info("请在左侧上传视频。注意：云端运行速度受服务器带宽限制。")
