import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import torch

from singer.model import SAMPLE_RATE_48K, YingSinger

# =============================================================================
# Model Initialization
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

try:
    singer = YingSinger(singer_path="ckpts", device=device)
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
    midi_file: str,
    lyrics: str,
    pitch_shift: int,
    cfg_strength: float,
    nfe_steps: int,
    sde_strength: float,
    seed: int,
) -> tuple[int, torch.Tensor]:
    """Run singing voice synthesis inference.

    Args:
        timbre_audio: Path to timbre reference audio.
        timbre_content: Text content of timbre reference.
        melody_audio: Path to melody reference audio.
        midi_file: Path to MIDI file (optional).
        lyrics: Target lyrics to synthesize.
        pitch_shift: Semitones to shift the melody key.
        cfg_strength: Classifier-free guidance strength.
        nfe_steps: Number of diffusion steps.
        sde_strength: Strength of SDE noise injection.
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
    if not melody_audio and not midi_file:
        raise gr.Error("Please upload melody reference audio or MIDI file. / 请上传旋律参考音频或 MIDI 文件")

    if midi_file:
        print(f"Using MIDI file for melody input: {midi_file}")
        gr.Info("Using MIDI file as melody input. / 使用 MIDI 文件作为旋律输入。")
        melody_audio = None
    else:
        print(f"Using audio file for melody input: {melody_audio}")
        gr.Info("Using audio file as melody input. / 使用音频文件提取旋律。")

    if not lyrics:
        raise gr.Error("Please enter lyrics. / 请输入歌词")

    try:
        print(f"Starting inference with seed: {seed}")
        gen_wav = singer.inference(
            timbre_audio_path=timbre_audio,
            timbre_audio_content=timbre_content,
            melody_audio_path=melody_audio,
            midi_file=midi_file,
            lyrics=lyrics,
            pitch_shift=int(pitch_shift),
            cfg_strength=float(cfg_strength),
            nfe_steps=int(nfe_steps),
            sde_strength=float(sde_strength),
            seed=int(seed) if seed is not None else 2025,
        )
        return (SAMPLE_RATE_48K, gen_wav.numpy().T)
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)} / 生成失败: {str(e)}")


demo_inputs = [
    {
        "timbre_audio": "resources/audios/female.wav",
        "timbre_content": "冰刀划的圈，圈起了谁改变。",
        "melody_audio": "resources/audios/female__Rnb_Funk__下等马_clip_001.wav",
        "midi_file": None,
        "lyrics": "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸屉揭盖。",
        "cfg_strength": 4.0,
        "nfe_steps": 32,
        "seed": 666,
        "sde_strength": 0.1,
        "pitch_shift": 0,
    },
    {
        "timbre_audio": "resources/audios/female.wav",
        "timbre_content": "冰刀划的圈，圈起了谁改变。",
        "melody_audio": None,
        "midi_file": "resources/audios/female__Rnb_Funk__下等马_clip_001.mid",
        "lyrics": "心敞开来，你泪水别太奇怪，会好的。没伤到未来，如果我爱得太深，请原谅我。疗伤十载，没出门没再恋爱，吃光回忆七八亩菜，练就这心坚强，莫以泪洗面哉。曙光压台，希望未衰，新生活谁还要来？速来领爱，别耽误我心扉敞开怀。",
        "cfg_strength": 4.0,
        "nfe_steps": 32,
        "seed": 666,
        "sde_strength": 0.1,
        "pitch_shift": 0,
    },
    {
        "timbre_audio": "resources/audios/male.wav",
        "timbre_content": "在爱的回归线，又期待相见。",
        "melody_audio": None,
        "midi_file": "resources/audios/female__Rnb_Funk__下等马_clip_001.mid",
        "lyrics": "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸屉揭盖。",
        "cfg_strength": 4.0,
        "nfe_steps": 32,
        "seed": 666,
        "sde_strength": 0.3,
        "pitch_shift": -9,
    },
]


with gr.Blocks(title="YingSinger WebUI") as app:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>YingMusic-Singer 零样本 歌声合成 & 编辑</h1>
        </div>
        """
    )

    gr.Markdown("### 1. 输入设置 （输入音频请使用干声、否则会影响使用效果）")
    with gr.Row():
        with gr.Column():
            timbre_audio = gr.Audio(label="音色参考音频（干声）", type="filepath")
            timbre_content = gr.Textbox(
                label="参考音频文本内容", placeholder="请输入参考音频中说/唱的文字内容，", lines=2
            )

        with gr.Column():
            with gr.Tabs():
                with gr.Tab("从音频提取旋律"):
                    melody_audio = gr.Audio(label="旋律参考音频（干声）", type="filepath")
                with gr.Tab("使用 MIDI 文件"):
                    midi_file = gr.File(label="MIDI 文件", file_types=[".mid", ".midi"])

            lyrics = gr.Textbox(label="目标歌词", placeholder="请输入想要合成的歌词", lines=2)

    with gr.Accordion("高级参数设置", open=True):
        with gr.Row():
            pitch_shift = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Pitch Shift (升降调)")
            cfg_strength = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="CFG Strength (引导强度)")
            nfe_steps = gr.Slider(minimum=10, maximum=200, value=32, step=1, label="NFE Steps (推理步数)")
            sde_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="SDE Strength")
            seed = gr.Number(value=666, label="随机种子 (Seed)", precision=0)

    submit_btn = gr.Button("开始生成", variant="primary")

    gr.Markdown("### 2. 生成结果（可以尝试不同的 SDE Strength 和 Seed 以获取更理想的效果）")
    output_audio = gr.Audio(label="合成音频", type="numpy")

    gr.Examples(
        examples=[
            [
                x["timbre_audio"],
                x["timbre_content"],
                x["melody_audio"],
                x["midi_file"],
                x["lyrics"],
                x["pitch_shift"],
                x["cfg_strength"],
                x["nfe_steps"],
                x["sde_strength"],
                x["seed"],
            ]
            for x in demo_inputs
        ],
        inputs=[
            timbre_audio,
            timbre_content,
            melody_audio,
            midi_file,
            lyrics,
            pitch_shift,
            cfg_strength,
            nfe_steps,
            sde_strength,
            seed,
        ],
        label="示例输入",
    )

    submit_btn.click(
        fn=run_inference,
        inputs=[
            timbre_audio,
            timbre_content,
            melody_audio,
            midi_file,
            lyrics,
            pitch_shift,
            cfg_strength,
            nfe_steps,
            sde_strength,
            seed,
        ],
        outputs=output_audio,
    )

if __name__ == "__main__":
    app.launch(allowed_paths=["."])
