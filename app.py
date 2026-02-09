import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import torch

from singer.model import SAMPLE_RATE_24K, YingSinger

# =============================================================================
# Model Initialization
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

try:
    singer = YingSinger(device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    singer = None


# =============================================================================
# Inference Function
# =============================================================================


def run_inference(
    timbre_audio: str,
    timbre_content: str,
    melody_audio: str,
    lyrics: str,
    cfg_strength: float,
    nfe_steps: int,
    seed: int,
) -> tuple[int, torch.Tensor]:
    """Run singing voice synthesis inference.

    Args:
        timbre_audio: Path to timbre reference audio.
        timbre_content: Text content of timbre reference.
        melody_audio: Path to melody reference audio.
        lyrics: Target lyrics to synthesize.
        cfg_strength: Classifier-free guidance strength.
        nfe_steps: Number of diffusion steps.
        seed: Random seed.

    Returns:
        Tuple of (sample_rate, audio_data).

    Raises:
        gr.Error: If model not loaded or required inputs missing.
    """
    if singer is None:
        raise gr.Error("Model not loaded. Please check the logs. / 模型未成功加载，请检查日志。")

    if not timbre_audio:
        raise gr.Error("Please upload timbre reference audio. / 请上传音色参考音频")
    if not timbre_content:
        raise gr.Error("Please enter reference audio text. / 请输入参考音频的文本内容")
    if not melody_audio:
        raise gr.Error("Please upload melody reference audio. / 请上传旋律参考音频")
    if not lyrics:
        raise gr.Error("Please enter lyrics. / 请输入歌词")

    try:
        print(f"Starting inference with seed: {seed}")
        gen_wav = singer.inference(
            timbre_audio_path=timbre_audio,
            timbre_audio_content=timbre_content,
            melody_audio_path=melody_audio,
            lyrics=lyrics,
            cfg_strength=float(cfg_strength),
            nfe_steps=int(nfe_steps),
            seed=int(seed) if seed is not None else 2025,
        )
        return (SAMPLE_RATE_24K, gen_wav.numpy())
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)} / 生成失败: {str(e)}")


demo_inputs = [
    {
        "timbre_audio": "resources/audios/zcd.wav",
        "timbre_content": "冰刀划的圈 圈起了谁改变",
        "melody_audio": "resources/audios/zcd.wav",
        "lyrics": "冰刃划的圆 围住了谁变迁",
        "cfg_strength": 3.0,
        "nfe_steps": 32,
        "seed": 2025,
    },
    {
        "timbre_audio": "resources/audios/zcd.wav",
        "timbre_content": "冰刀划的圈 圈起了谁改变",
        "melody_audio": "resources/audios/p_1.wav",
        "lyrics": "你说 你演错了剧本 陪尽了天真心真",
        "cfg_strength": 3.0,
        "nfe_steps": 32,
        "seed": 2025,
    },
    {
        "timbre_audio": "resources/audios/p_1.wav",
        "timbre_content": "你说 你演错了剧本 陪尽了天真心真",
        "melody_audio": "resources/audios/zcd.wav",
        "lyrics": "寒刀画的环 包住了谁转变",
        "cfg_strength": 3.0,
        "nfe_steps": 32,
        "seed": 2025,
    },
]


with gr.Blocks(title="YingSinger WebUI") as app:
    # <p>
    #     当前模型为 <b style="color: #ff4b4b;">beta</b> 版本，仅支持 <b>中文 pop music 与较低的音质</b><br>
    #     v1 版本 <b>(多语言 & 更高音质 & 更好的泛化性)</b> 将在 2025 年底之前推出，敬请期待...
    # </p>
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>YingMusic-Singer 零样本 歌声合成 & 编辑</h1>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. 输入设置 （输入音频请使用干声、否则会影响使用效果）")
            timbre_audio = gr.Audio(label="音色参考音频（干声） -- 音色参考音频不应超过 10s", type="filepath")
            timbre_content = gr.Textbox(
                label="参考音频文本内容", placeholder="请输入参考音频中说/唱的文字内容，", lines=2
            )

            melody_audio = gr.Audio(label="旋律参考音频（干声） -- 旋律参考音频不应超过 20s", type="filepath")
            lyrics = gr.Textbox(label="目标歌词", placeholder="请输入想要合成的歌词", lines=2)

            with gr.Accordion("高级参数设置", open=False):
                cfg_strength = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="CFG Strength (引导强度)"
                )
                nfe_steps = gr.Slider(minimum=10, maximum=200, value=32, step=1, label="NFE Steps (推理步数)")
                seed = gr.Number(value=2025, label="随机种子 (Seed)", precision=0)

            submit_btn = gr.Button("开始生成", variant="primary")

        with gr.Column():
            gr.Markdown("### 2. 生成结果")
            output_audio = gr.Audio(label="合成音频", type="numpy")

            gr.Examples(
                examples=[
                    [
                        x["timbre_audio"],
                        x["timbre_content"],
                        x["melody_audio"],
                        x["lyrics"],
                        x["cfg_strength"],
                        x["nfe_steps"],
                        x["seed"],
                    ]
                    for x in demo_inputs
                ],
                inputs=[
                    timbre_audio,
                    timbre_content,
                    melody_audio,
                    lyrics,
                    cfg_strength,
                    nfe_steps,
                    seed,
                ],
                label="示例输入",
            )

    submit_btn.click(
        fn=run_inference,
        inputs=[timbre_audio, timbre_content, melody_audio, lyrics, cfg_strength, nfe_steps, seed],
        outputs=output_audio,
    )

if __name__ == "__main__":
    app.launch()
