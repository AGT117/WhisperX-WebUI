import json
import time
import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog
import gradio as gr
from pathlib import Path
from src.core.engine import FullPipelineEngine
from src.core.utils import generate_srt, format_transcript_for_display
from config.settings import OUTPUT_DIR, HF_TOKEN

# 引擎实例化
engine = FullPipelineEngine()

def process_batch_task(
    file_paths, model_size, lang, 
    enable_diar, min_spk, max_spk, 
    vad_onset, initial_prompt, compute_type, enable_demucs, 
    release_memory, export_srt,
    custom_output_path
):
    """WebUI 批处理回调函数"""
    if not file_paths:
        yield "警告: 请先上传文件"
        return
    
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    # 路径解析逻辑
    try:
        if custom_output_path and custom_output_path.strip():
            save_dir = Path(custom_output_path.strip())
        else:
            save_dir = OUTPUT_DIR
        
        # 确保目标目录存在，如果不存在则自动创建
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"已自动创建输出目录: {save_dir}")
    except Exception as e:
        yield f"错误: 自定义路径无效 ({str(e)})，请检查路径格式。"
        return

    total_files = len(file_paths)
    log_buffer = ""      

    min_spk = int(min_spk) if min_spk else None
    max_spk = int(max_spk) if max_spk else None
    
    initial_prompt = initial_prompt.strip() if initial_prompt else None

    # 批处理循环
    for i, file_path in enumerate(file_paths):
        current_index = i + 1
        input_path = Path(file_path)
        file_stem = input_path.stem
        
        mode_info = "[人声分离模式]" if enable_demucs else "[标准转录模式]"
        status_msg = f"[进度: {current_index}/{total_files}] 正在处理: {file_stem} ... {mode_info} (VAD阈值: {vad_onset})"
        current_log = status_msg + "\n" + log_buffer
        yield current_log

        start_time = time.time()
        
        try:
            # 执行核心管道
            segments, status = engine.run_pipeline(
                audio_path=file_path, 
                model_size=model_size, 
                lang=lang, 
                enable_diarization=enable_diar, 
                min_speakers=min_spk, 
                max_speakers=max_spk,
                vad_onset=vad_onset,        
                initial_prompt=initial_prompt,
                compute_type=compute_type,
                enable_demucs=enable_demucs 
            )

            if "Error" in status or "Failed" in status:
                raise RuntimeError(status)

            # 结果持久化
            json_path = save_dir / f"{file_stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

            if export_srt:
                srt_path = save_dir / f"{file_stem}.srt"
                srt_content = generate_srt(segments)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

            # 更新 UI 日志
            duration = time.time() - start_time
            display_text = format_transcript_for_display(segments)
            
            file_log = (
                f"[完成: {current_index}/{total_files}] {file_stem} (耗时: {duration:.2f}s)\n"
                f"输出路径: {save_dir}\n"
                f"{'-'*30}\n"
                f"{display_text}\n"
                f"{'='*30}\n\n"
            )
            log_buffer = file_log + log_buffer
            yield log_buffer

        except Exception as e:
            error_log = (
                f"[失败: {current_index}/{total_files}] {file_stem}\n"
                f"原因: {str(e)}\n"
                f"{'='*30}\n\n"
            )
            log_buffer = error_log + log_buffer
            print(f"Task Failed: {e}")
            yield log_buffer

    if release_memory:
        yield "正在执行资源释放...\n" + log_buffer
        engine.unload_all()
        yield "所有任务执行完毕，显存已释放。\n" + log_buffer
    else:
        yield "所有任务执行完毕。\n" + log_buffer

def create_ui():
    """Gradio 界面构建"""
    with gr.Blocks(title="WhisperX WebUI By AGT") as app:
        gr.Markdown("## WhisperX WebUI By AGT")
        
        if not HF_TOKEN:
            gr.Warning("警告: 未检测到 HF_TOKEN，说话人分离功能将不可用。")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="文件批处理输入",
                    type="filepath",
                    file_types=["audio", "video"],
                    file_count="multiple",
                    height=120
                )
                
                with gr.Tabs():
                    with gr.TabItem("基础配置"):
                        model_sel = gr.Dropdown(
                            ["base", "small", "medium", "large-v2", "large-v3"], 
                            value="large-v3", 
                            label="Whisper 模型架构",
                            info="模型越大识别精度越高，但显存占用和推理耗时也会显著增加。8GB 显存及以上推荐使用 large-v3。"
                        )
                        lang_sel = gr.Dropdown(
                            [None, "zh", "en", "ja", "ko"], 
                            value=None, 
                            label="源语言设置",
                            info="若不确定，请保持为空以启用自动检测。指定语言可略微提升准确率。项目目前仅对英文（en）做了支持，不过西方语系问题应该都不大，但中日韩的效果可能效果奇差，别选！！！"
                        )
                        
                        gr.Markdown("---")
                        enable_demucs = gr.Checkbox(
                            label="启用人声分离预处理 (BS-RoFormer)", 
                            value=False,
                            info="在识别前将人声从背景音乐或噪音中分离。适用于歌曲或高噪环境，会增加总处理时间，但对于提高识别率很有帮助。"
                        )
                        
                        prompt_input = gr.Textbox(
                            label="上下文提示词 (Prompt)",
                            info="提供给模型的风格引导或专有名词参考（并非指令）。例如输入歌词的前几句，或指定标点符号风格。",
                            value="Hello, welcome to my music world. Please add punctuation to the lyrics.",
                            placeholder="请输入提示词...",
                            lines=3,
                            interactive=True 
                        )

                    with gr.TabItem("推理参数"):
                        compute_type_sel = gr.Dropdown(
                            ["int8", "float16", "float32"],
                            value="float16",
                            label="量化精度",
                            info="float16 为 GPU 标准精度；int8 可大幅降低显存占用但可能轻微损失精度；float32 仅用于 CPU 模式。"
                        )
                        vad_onset_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.35, step=0.05,
                            label="VAD 触发阈值 (Onset)",
                            info="语音活动检测的灵敏度。数值越高越严格（减少幻觉），数值越低越灵敏（保留更多细节）。默认 0.35。"
                        )
                        
                    with gr.TabItem("说话人区分"):
                        enable_diar = gr.Checkbox(
                            label="启用说话人聚类 (Diarization)", 
                            value=False,
                            info="识别并区分音频中的不同说话人。需要有效的 HuggingFace Token。"
                        )
                        with gr.Row():
                            min_spk = gr.Number(label="最小说话人数", value=1, precision=0)
                            max_spk = gr.Number(label="最大说话人数", value=5, precision=0)

                with gr.Accordion("输出与资源管理", open=False):
                    export_srt = gr.Checkbox(label="同步导出 SRT 字幕文件", value=True)
                    release_mem = gr.Checkbox(
                        label="任务完成后释放显存", 
                        value=True,
                        info="任务结束后卸载模型并执行垃圾回收，释放 GPU 资源给其他应用。取消勾选能够提升批处理时候的效率。"
                    )

                with gr.Row():
                    btn_submit = gr.Button("开始处理任务", variant="primary", scale=2)
                    btn_stop = gr.Button("终止任务", variant="stop", scale=1)

            with gr.Column(scale=2):
                text_out = gr.TextArea(
                    label="系统运行日志", 
                    lines=30,
                    placeholder="任务进度与详细日志将显示在此处..."
                )
                
                with gr.Row(variant="panel"):
                    # 路径显示框 (scale=5 占据主要宽度)
                    output_dir_input = gr.Textbox(
                        label="输出目录 (Output Directory)", 
                        value=str(OUTPUT_DIR),
                        info="选择或输入结果保存路径",
                        scale=5,
                        max_lines=1,
                        interactive=True
                    )
                    # 浏览按钮 (scale=1)
                    btn_browse = gr.Button("选择文件夹", scale=1)
                    # 打开按钮 (scale=1)
                    btn_open = gr.Button("打开", scale=1, variant="secondary")

                # 1. 浏览文件夹逻辑 (调用 Tkinter)
                def browse_folder_action():
                    try:
                        # 创建一个隐藏的 Tk 窗口作为根
                        root = tk.Tk()
                        root.withdraw() # 隐藏主窗口
                        root.attributes('-topmost', True) # 确保弹窗在最顶层，防止被浏览器遮挡
                        
                        selected_path = filedialog.askdirectory()
                        
                        root.destroy() # 销毁 Tk 实例
                        
                        if selected_path:
                            return str(Path(selected_path))
                        else:
                            return gr.update() # 如果取消，保持原值
                    except Exception as e:
                        print(f"Tkinter dialog error: {e}")
                        return gr.update()

                # 2. 打开文件夹逻辑
                def open_folder_action(custom_path):
                    path = custom_path.strip() if custom_path else str(OUTPUT_DIR)
                    path_obj = Path(path)
                    
                    if not path_obj.exists():
                        try:
                            path_obj.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            return 
                    
                    if sys.platform == "win32":
                        os.startfile(path)
                    elif sys.platform == "darwin":
                        subprocess.call(["open", path])
                    else:
                        subprocess.call(["xdg-open", path])

                btn_browse.click(browse_folder_action, inputs=None, outputs=output_dir_input)
                btn_open.click(open_folder_action, inputs=[output_dir_input], outputs=None)

        process_event = btn_submit.click(
            process_batch_task, 
            inputs=[
                file_input, model_sel, lang_sel, 
                enable_diar, min_spk, max_spk, 
                vad_onset_slider, prompt_input, compute_type_sel, 
                enable_demucs, 
                release_mem, export_srt,
                output_dir_input # 将选择的路径传给后端
            ], 
            outputs=[text_out], 
            show_progress="minimal" 
        )
        
        btn_stop.click(fn=None, inputs=None, outputs=None, cancels=[process_event])
    
    return app