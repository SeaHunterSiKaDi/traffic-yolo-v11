if uploaded_file is not None:
    # 直接读取上传文件的二进制内容到内存
    video_bytes = uploaded_file.read()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st_summary = st.empty()
        st_chart = st.empty()

    try:
        # 使用 imageio 的 <video0> 模式或直接传入字节流
        # 这里我们改用一个更稳妥的办法：直接用字节流初始化 reader
        reader = imageio.get_reader(video_bytes, format='mp4')
        
        counts_history = []
        
        # 为了速度和稳定性，每 10 帧处理一次
        for i, frame in enumerate(reader):
            if i % 10 != 0: continue
            
            # 预测与渲染
            results = model.predict(frame, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()
            
            # 车辆计数逻辑
            current_count = sum(1 for box in results[0].boxes if model.names[int(box.cls[0])] in ['car', 'truck', 'bus'])
            counts_history.append(current_count)
            
            # 显示画面
            img = Image.fromarray(annotated_frame)
            st_frame.image(img, caption="AI 实时监测中...", use_container_width=True)
            
            # 数据可视化
            st_summary.metric("当前车辆数", current_count)
            if len(counts_history) > 1:
                st_chart.line_chart(counts_history[-30:]) 
                
        reader.close()
        st.success("🎉 视频全部分析完成！")
        
    except Exception as e:
        st.error(f"分析出错：{e}")
        st.info("提示：请确保上传的是标准 mp4 格式视频。")
