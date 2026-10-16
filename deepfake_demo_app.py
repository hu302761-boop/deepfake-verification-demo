#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# 深度伪造视频核验 Demo - 可交互分析展示
# 面向 AI 产品经理岗位的求职 Demo
# ============================================================

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os, base64, json
from pathlib import Path
from io import BytesIO

# ---- 第1部分：读取分析结果，准备展示数据 ----

# 把PNG转成base64，方便在网页中显示
def img_to_html(path, width="100%"):
    """读取图片并转成HTML能显示的base64格式"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="width:{width};border-radius:8px;">'
    return "<p>图片缺失</p>"

PROJECT_DIR = Path(__file__).resolve().parent
FIG_DIR = PROJECT_DIR / "results" / "figures"
RERUN_SUMMARY_PATH = PROJECT_DIR / "results" / "calculation_rerun_v1" / "summary.json"


def load_calculation_summary():
    """只读取确定性复跑的描述性结果，避免 Demo 与论文证据版本脱节。"""
    with RERUN_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


CALCULATION_SUMMARY = load_calculation_summary()
VISUAL = CALCULATION_SUMMARY["FakeAVCeleb_visual"]
LANDMARK = CALCULATION_SUMMARY["FaceForensics_landmarks_Deepfakes"]
BLOCK_BOUNDARY = CALCULATION_SUMMARY["FaceForensics_block_boundary_Deepfakes"]
AUDIO = CALCULATION_SUMMARY["FakeAVCeleb_audio"]


def percentage(value):
    """保留必要的小数位，例如 88.75%。"""
    return f"{value:g}%"

# ---- 第2部分：定义每个分析层级的展示内容 ----

PHYSICAL_CUES = [
    "面部边缘或发丝细节异常平滑",
    "皮肤纹理与自然噪声变化明显不足",
    "面部关键点出现不自然的微小漂移",
]
MULTIMODAL_CUES = [
    "语音的自然停顿、气息或背景变化不足",
    "唇形、发音与语音节奏存在可见不一致（需排除配音与剪辑）",
    "环境声与画面环境不匹配",
]
LOGIC_CUES = [
    "人物、事件、时间或地点信息无法相互印证",
    "光影、反射或镜头内物理关系不一致",
    "来源、发布账号或上下文缺少可核验依据",
]


def build_diagnosis(physical, multimodal, logical):
    """把用户观察转为可解释的核验建议，而非输出未经验证的真假判定。"""
    physical = physical or []
    multimodal = multimodal or []
    logical = logical or []
    layers = []
    if physical:
        layers.append(("物理层", physical, "暂停结论，回看原清晰度版本，并排除低码率、滤镜和视频通话压缩影响。"))
    if multimodal:
        layers.append(("多模态层", multimodal, "分离检查音频与画面；核对音画同步、环境声及完整上下文。"))
    if logical:
        layers.append(("逻辑层", logical, "追溯首发来源、发布日期、地点与权威信息，避免只凭画面直觉判断。"))

    if not layers:
        return """
        <h3>尚未形成可解释线索</h3>
        <p>当前不应据此认定真实或伪造。请至少从一个层级记录可观察线索，再进行来源核验。</p>
        """

    entries = "".join(
        f"<li><b>{name}</b>：{'；'.join(cues)}<br><span style='color:#555'>下一步：{action}</span></li>"
        for name, cues, action in layers
    )
    if len(layers) >= 2:
        conclusion = "已出现跨层线索：建议优先做原始来源与完整上下文核验，再由人工给出判断。"
    else:
        conclusion = "目前只有单层线索：它只能触发复核，不能单独作为真假结论。"
    return f"""
    <h3>可解释诊断摘要</h3>
    <p><b>{conclusion}</b></p>
    <ol>{entries}</ol>
    <p style='color:#666'>提示：本工作台不输出“AI 判伪分数”，避免把压缩、滤镜、平台标识或单一异常误当成深度伪造证据。</p>
    """

def show_physical_layer():
    """物理层计算线索展示。"""
    edge = percentage(VISUAL["edge"]["real_higher_pct"])
    noise = percentage(VISUAL["noise"]["real_higher_pct"])
    landmark = percentage(LANDMARK["real_higher_pct"])
    block = percentage(BLOCK_BOUNDARY["real_higher_pct"])
    html = f"""
    <h2>🔬 物理层分析</h2>
    <p><strong>计算验证的候选线索：</strong>在固定配对与处理条件下，伪造样本更常呈现高频细节和自然微变动的缺失；这不是单一判伪规则，也不是受众实验结果。</p>
    <table style="width:100%;border-collapse:collapse;">
    <tr style="background:#2196F3;color:white;">
        <th>特征</th><th>结果</th><th>可靠性</th><th>解读</th>
    </tr>
    <tr><td>面部边缘密度</td><td>{edge} 真实&gt;伪造（n={VISUAL['edge']['n']}）</td><td>候选</td><td>该配对中真实更常保留边缘细节</td></tr>
    <tr><td>像素噪声分布</td><td>{noise} 真实&gt;伪造（n={VISUAL['noise']['n']}）</td><td>候选</td><td>该配对中真实残差波动更高</td></tr>
    <tr><td>关键点自然位移</td><td>{landmark} 真实&gt;伪造（n={LANDMARK['n']}）</td><td>候选</td><td>该配对中真实更常保留自然微变动</td></tr>
    <tr><td>块边界线索</td><td>{block} 真实&gt;伪造（n={BLOCK_BOUNDARY['n']}）</td><td>辅助</td><td>仅限 FaceForensics++ Deepfakes 的候选方向性证据</td></tr>
    <tr><td>眼神光缺失</td><td>工具受限</td><td>—</td><td>需更高精度检测</td></tr>
    </table>
    """ + img_to_html(FIG_DIR / "thesis_calculation_validation" / "图2-1_物理层候选线索_方向性证据.png")
    return html

def show_multimodal_layer():
    """多模态计算线索展示。"""
    zcr = percentage(AUDIO["zcr"]["real_higher_pct"])
    spectral = percentage(AUDIO["spectral_variance"]["real_higher_pct"])
    rms = percentage(AUDIO["rms"]["real_higher_pct"])
    html = f"""
    <h2>🎵 多模态分析</h2>
    <p><strong>计算验证的候选线索：</strong>在 FakeAVCeleb 的固定配对中，TTS 音频更常缺少自然的声学变化；需与画面及内容线索交叉验证，不可据此直接判伪。</p>
    <table style="width:100%;border-collapse:collapse;">
    <tr style="background:#E91E63;color:white;">
        <th>特征</th><th>结果</th><th>可靠性</th><th>解读</th>
    </tr>
    <tr><td>音频过零率(ZCR)</td><td>{zcr} 真实&gt;伪造（n={AUDIO['zcr']['n']}）</td><td>候选</td><td>该配对中自然语音变化更丰富</td></tr>
    <tr><td>频谱方差</td><td>{spectral} 真实&gt;伪造（n={AUDIO['spectral_variance']['n']}）</td><td>候选</td><td>该配对中自然声学变化更丰富</td></tr>
    <tr><td>音频 RMS 能量</td><td>{rms} 真实&gt;伪造（n={AUDIO['rms']['n']}）</td><td>候选</td><td>该配对中自然响度变化更丰富</td></tr>
    <tr><td>唇形-语音同步</td><td>40% 支持预期（n=48）</td><td>探索性</td><td>当前方法证据不足，只作辅助复核</td></tr>
    </table>
    """ + img_to_html(FIG_DIR / "thesis_calculation_validation" / "图2-2_多模态候选线索_方向性证据.png")
    return html

def show_summary():
    """总览。"""
    html = f"""
    <h2>📊 三层级检测框架总览</h2>
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
    <tr style="background:#4CAF50;color:white;">
        <th>层级</th><th>完成特征</th><th>最强特征</th><th>核心发现</th>
    </tr>
    <tr><td><b>物理层</b></td><td>4个候选特征</td><td>关键点自然位移 {percentage(LANDMARK['real_higher_pct'])}</td><td>高频细节或自然微变动可能缺失</td></tr>
    <tr><td><b>多模态</b></td><td>4个候选特征</td><td>过零率 {percentage(AUDIO['zcr']['real_higher_pct'])}</td><td>合成语音可能缺乏自然变化</td></tr>
    <tr><td><b>逻辑层</b></td><td>核验流程（待材料预实验）</td><td>—</td><td>人物、事件、来源与时间线的交叉核验</td></tr>
    </table>
    <h3>论文核心结论</h3>
    <p>本项目建议从<strong>“寻找单一伪造痕迹”转向“识别真实特征是否缺失，并进行多线索交叉核验”</strong>。表中比例来自确定性计算复跑，属于候选方向性证据，不是受众实验或产品准确率。</p>
    """ + img_to_html(FIG_DIR / "thesis_calculation_validation" / "图2-1_物理层候选线索_方向性证据.png") + img_to_html(FIG_DIR / "thesis_calculation_validation" / "图2-2_多模态候选线索_方向性证据.png")
    return html

# ---- 第3部分：构建Gradio界面 ----

with gr.Blocks(title="深度伪造视频可解释核验工作台 - 胡静", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 深度伪造视频可解释核验工作台
    ## 面向公众判断的三阶分析框架
    
    **研究者：胡静** | **目标岗位：AI 产品经理** | 数据集：FakeAVCeleb + FaceForensics++

    > 这是研究型产品原型：展示“线索—限制—下一步核验”的决策路径，而非对任意视频输出黑箱真假分数。受众预实验与正式随机对照实验尚未实施。
    
    ---
    """)
    
    with gr.Tabs():
        with gr.Tab("📊 总览"):
            gr.HTML(show_summary)
        
        with gr.Tab("🔬 物理层"):
            gr.HTML(show_physical_layer)
        
        with gr.Tab("🎵 多模态"):
            gr.HTML(show_multimodal_layer)

        with gr.Tab("🧭 可解释诊断工作台"):
            gr.Markdown("""
            ### 从“给分数”转向“给核验路径”

            勾选你在一段视频中实际观察到的线索。系统会显示应当如何复核；它不会在证据不足时给出真假结论。
            """)
            with gr.Row():
                with gr.Column():
                    physical_input = gr.CheckboxGroup(PHYSICAL_CUES, label="物理层：像素与面部细节")
                with gr.Column():
                    multimodal_input = gr.CheckboxGroup(MULTIMODAL_CUES, label="多模态层：声音与画面")
                with gr.Column():
                    logical_input = gr.CheckboxGroup(LOGIC_CUES, label="逻辑层：内容与来源")
            diagnose_button = gr.Button("生成核验路径", variant="primary")
            diagnosis_output = gr.HTML("<p>选择线索后，生成可解释的复核建议。</p>")
            diagnose_button.click(
                build_diagnosis,
                inputs=[physical_input, multimodal_input, logical_input],
                outputs=diagnosis_output,
            )
        
        with gr.Tab("ℹ️ 关于"):
            gr.Markdown("""
            ### 关于本项目
            
            本 Demo 将深度伪造研究转译为面向普通用户的产品交互：用户不是获得一个无法解释的真假分数，而是获得可追溯的线索、限制和下一步核验建议。
            
            - **原型界面：** Python + Gradio
            - **离线计算验证：** OpenCV、Librosa；FakeAVCeleb v1.2、FaceForensics++ c23
            - **当前产品状态：** 本地原型已验证运行；未对外部署，不声称具备任意视频的自动检测准确率
            
            ### 三层级分析
            1. **物理层** — 像素级噪声、边缘密度、关键点稳定性
            2. **多模态** — 音频过零率、频谱方差、唇形同步
            3. **逻辑层** — 人物、事件、来源与时间线的交叉核验

            ### 产品原则
            - 不把单一视觉异常或平台标识当作深度伪造结论
            - 输出可追溯的线索、限制与下一步核验路径
            - 将“直接真伪提示”“来源标识”和“内容异常”分开处理
            - 带 AI 标识、水印、台标或字幕的素材可保留为候选，但不能把“看见标识”误当成识别视频内在痕迹
            
            ### 联系方式
            胡静 | 求职方向：AI产品经理
            """)

# ---- 第4部分：启动 ----
if __name__ == "__main__":
    # 默认固定本地端口；验收时可通过 DEMO_PORT 使用临时端口，仍仅监听本机。
    demo_port = int(os.environ.get("DEMO_PORT", "7861"))
    demo.launch(share=False, server_name="127.0.0.1", server_port=demo_port)
    print(f"✅ Demo已启动，浏览器打开 http://127.0.0.1:{demo_port}")


# In[ ]:
