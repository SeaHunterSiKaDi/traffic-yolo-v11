import streamlit as st
from ultralytics import YOLO
import imageio
from PIL import Image
import numpy as np
import tempfile
import os

# 强制禁用所有图形显示后端，防止底层崩溃
os.environ["QT_QPA_PLATFORM"] = "offscreen"

st.set_page_config(page_title="AI 交通分析系统", layout="wide")

@st.cache_resource
def load_model():
    # 加载最轻量级的模型
    return YOLO('yolo11n.pt')

model = load_model()

st.title("🚗 在线智能交通流监测平台")
st.info("当前运行模式：云端稳定兼容模式 (无需 OpenCV)")

uploaded_file = st.sidebar.file_uploader("第一步：上传监控视频", type=['mp4', 'avi'])

if uploaded_file is not None:
    # 1. 处理临时文件
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st_summary = st.empty()
        st_chart = st.empty()

    # 2. 使用 imageio 读取视频（绕过 cv2.VideoCapture）
    try:
        reader = imageio.get_reader(tfile.name,  'ffmpeg')
        fps = reader.get_meta_data().get('fps', 30)
        
        counts_history = []
        
        # 抽帧处理，防止云端内存溢出（每 5 帧处理 1 帧）
        for i, frame in enumerate(reader):
            if i % 5 != 0: continue
            
            # YOLO 预测
            results = model.predict(frame, conf=0.25, verbose=False)
            
            # 渲染结果（直接获取渲染后的 numpy 数组）
            annotated_frame = results[0].plot()
            
            # 统计车辆
            current_count = 0
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if model.names[cls_id] in ['car', 'truck', 'bus']:
                    current_count += 1
            
            counts_history.append(current_count)
            
            # 转化为 PIL 图片展示
            img = Image.fromarray(annotated_frame)
            st_frame.image(img, caption=f"实时检测中 - 帧数: {i}", use_container_width=True)
            
            # 更新右侧数据
            st_summary.metric("当前车辆数", current_count)
            if len(counts_history) > 1:
                st_chart.line_chart(counts_history[-20:]) # 显示最近 20 帧的波动
                
        reader.close()
        st.success("视频处理完毕！")
        
    except Exception as e:
        st.error(f"处理失败: {e}")
    finally:
        if os.path.exists(tfile.name):
            os.remove(tfile.name)

else:
    st.write("请在左侧侧边栏上传视频文件（支持 mp4/avi）。")
