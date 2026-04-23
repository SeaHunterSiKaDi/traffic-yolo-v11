import streamlit as st
import os

# 1. 基础配置与环境兼容
os.environ["QT_QPA_PLATFORM"] = "offscreen"
st.set_page_config(page_title="AI 交通分析系统", layout="wide")

# 只有在代码最顶层成功导入所需的库，后面才不会报错
try:
    from ultralytics import YOLO
    import imageio
    from PIL import Image
    import numpy as np
except ImportError as e:
    st.error(f"核心库导入失败，请检查 requirements.txt: {e}")
    st.stop()

# 2. 模型加载（带缓存）
@st.cache_resource
def load_model():
    # 使用最小的 n 模型，确保云端不爆内存
    return YOLO('yolo11n.pt')

model = load_model()

# 3. 界面布局
st.title("🚗 在线智能交通流监测平台")
st.info("当前模式：内存流处理模式 (跳过临时文件)")

# 4. 定义变量：这是解决你刚才报错的关键
# 必须先用 streamlit 组件定义出 uploaded_file
uploaded_file = st.sidebar.file_uploader("第一步：上传监控视频", type=['mp4', 'avi'])

# 5. 业务逻辑：当文件被上传后才执行
if uploaded_file is not None:
    # 读取视频字节流
    video_bytes = uploaded_file.read()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st_summary = st.empty()
        st_chart = st.empty()

    try:
        # 使用 imageio 直接从内存字节读取
        reader = imageio.get_reader(video_bytes, format='mp4')
        counts_history = []
        
        # 抽帧处理：云端服务器性能有限，每 10 帧处理 1 帧
        for i, frame in enumerate(reader):
            if i % 10 != 0: continue
            
            # YOLO 执行预测
            results = model.predict(frame, conf=0.25, verbose=False)
            
            # 渲染检测框
            annotated_frame = results[0].plot()
            
            # 统计特定类别：汽车、卡车、公交
            current_count = sum(1 for box in results[0].boxes if model.names[int(box.cls[0])] in ['car', 'truck', 'bus'])
            counts_history.append(current_count)
            
            # 将渲染后的图片显示在网页上
            img = Image.fromarray(annotated_frame)
            st_frame.image(img, caption="AI 实时监测中...", use_container_width=True)
            
            # 更新右侧统计数据
            st_summary.metric("当前画面车辆数", current_count)
            if len(counts_history) > 1:
                st_chart.line_chart(counts_history[-30:]) 
                
        reader.close()
        st.success("🎉 分析完成！")
        
    except Exception as e:
        st.error(f"处理视频时发生错误: {e}")
else:
    st.write("👈 请先在左侧侧边栏上传一段交通监控视频。")
