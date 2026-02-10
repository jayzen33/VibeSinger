# YingMusic-Singer: Zero-shot Singing Voice Synthesis and Editing with Annotation-free Melody Guidance

<!-- --- -->

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-YingMusic--Singer--Beta-blue)](https://arxiv.org/pdf/2512.04779)
[![Demo](https://img.shields.io/badge/🎶%20Demo-YingMusic--Singer-red)](https://giantailab.github.io/YingMusic-Singer-demo/)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-YingMusic--Singer-yellow)](https://huggingface.co/GiantAILab/YingMusic-Singer)
[![ModelScope](https://img.shields.io/badge/🔮%20ModelScope-YingMusic--Singer-purple)](https://www.modelscope.cn/models/giantailab/YingMusic-Singer/)

</div>

> **Note**: The beta version is deprecated. If you need to use it, please switch to the `beta` branch.

---

## Overview ✨

<!-- <p align="center">
  <img src="figs/head.jpeg" width="720" alt="pipeline">
</p> -->

**YingMusic-Singer** is a unified framework for **Zero-shot Singing Voice Synthesis (SVS) and Editing**, driven by **Annotation-free Melody Guidance**. Addressing the scalability challenges of real-world applications, our system eliminates the reliance on costly phoneme-level alignment and manual melody annotations. It enables **arbitrary lyrics to be synthesized or edited with any reference melody** in a zero-shot manner.

<!-- Our approach leverages a **Diffusion Transformer (DiT)** based generative model, incorporating a pre-trained melody extraction module to derive MIDI information directly from reference audio. By introducing a structured guidance mechanism and employing **Flow-GRPO reinforcement learning**, we achieve superior pronunciation clarity, melodic accuracy, and musicality without requiring fine-grained alignment. -->

### 🔧 Key Features

- **Unified Synthesis & Editing**: Seamlessly integrates zero-shot synthesis and editing within a single framework.
- **Annotation-free Melody Guidance**: Automatically extracts melody from reference audio, eliminating the need for manual MIDI or phoneme alignment.
- **Zero-Shot Capabilities**: Generates high-quality singing voices from arbitrary lyrics and melodies without fine-tuning on the target voice.
- **Flexible Melody Input**: Accepts both reference audio and direct MIDI files as melody inputs for enhanced versatility.
  <!-- - **Flow-GRPO Reinforcement Learning**: Optimizes pronunciation, melodic accuracy, and musicality via multi-objective rewards. -->
  <!-- - **Structured Guidance Mechanism**: Enhances melodic stability and coherence using similarity distribution constraints. -->
  <!-- - **Robust Generalization**: Outperforms existing methods in zero-shot synthesis and lyric replacement scenarios. -->

---

<p align="center">
  <img src="resources/imgs/main_v1.svg" width="720" alt="pipeline">
</p>

---

## News & Updates 🗞️

- **2025-11-27**: Released the technical report.
- **2025-11-26**: Released the beta version's inference code and model checkpoints.
- **2026-02-09**: Released V1 inference code and checkpoints, featuring bilingual support (Chinese & English), significantly improved audio quality, and enhanced generalization.
- **2026-02-09**: Released the V1 version's [demo pages](https://giantailab.github.io/YingMusic-Singer-demo/).

---

## Roadmap & TODO 🗺️

- [x] Release beta version inference code and model checkpoints (currently supports Chinese & lower audio quality).
- [x] Release V1 Version: Support for Chinese & English singing with higher audio quality and better generalization.

---

## Installation 🛠️

```bash
git clone https://github.com/GiantAILab/YingMusic-Singer.git
cd YingMusic-Singer

conda create -n singer python=3.12
conda activate singer

# Install PyTorch with CUDA 12.6 support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Install flash_attn
pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip3 install -r requirements.txt

unset PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

---

## Quick Start 🚀

Download model checkpoints from [huggingface](https://huggingface.co/GiantAILab/YingMusic-Singer) or [modelscope](https://www.modelscope.cn/models/giantailab/YingMusic-Singer/)

> **Note**: If the pitch range of the reference audio differs significantly from the target melody (MIDI or audio), manually adjusting the key is recommended for optimal results.

```bash
# Please keep the prompt audio duration is around 5-7 seconds, and the total duration does not exceed 45 seconds.

# infer from MIDI file
python src/singer/model.py --timbre_audio_path resources/audios/0000.wav \
    --timbre_audio_content "在爱的回归线 又期待相见" \
    --midi_file "resources/audios/female__Rnb_Funk__下等马_clip_001.mid" \
    --lyrics "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸屉揭盖。" \
    --out_path "outputs/test_yingsinger_zs.wav" \
    --cfg_strength 4.0 \
    --nfe_steps 64 \
    --pitch_shift -4

# infer from melody audio
python src/singer/model.py --timbre_audio_path resources/audios/0000.wav \
    --timbre_audio_content "在爱的回归线 又期待相见" \
    --melody_audio_path "resources/audios/female__Rnb_Funk__下等马_clip_001.wav" \
    --lyrics "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸屉揭盖。" \
    --out_path "outputs/test_yingsinger_zs.wav" \
    --cfg_strength 4.0 \
    --nfe_steps 64 \
    --pitch_shift -4
```

<!-- ### 2. Singing Voice Editing

```bash
# Please keep the prompt audio duration is around 5-7 seconds, and the total duration does not exceed 45 seconds.
python src/singer/model.py --ckpt_path "ckpt_path" \
    --timbre_audio_path "resources/audios/mxsf.wav" \
    --timbre_audio_content "你说 你爱了不该爱的人 你的心中满是伤痕" \
    --melody_audio_path "resources/audios/mxsf.wav" \
    --lyrics "你说 你演错了剧本 赔尽了天真心真" \
    --out_path "outputs/test_yingsinger.wav" \
    --cfg_strength 3.0 \
    --nfe_steps 100
``` -->

---

## Acknowledgements 🙏

We would like to express our gratitude to the following projects for their contributions:

- **[SOME](https://github.com/openvpi/SOME)**: For the Singing-Oriented MIDI Extractor, which we use as our melody extractor.
- **[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)**: For the RMVPE model used for pitch extraction.
- **[SongBloom](https://github.com/tencent-ailab/SongBloom)**: For the [Stable Audio VAE](https://github.com/Stability-AI/stable-audio-tools) implementation.

## Citation 🧾

If you use YingMusic-Singer for research, please cite:

```
@article{zheng2025yingmusicsinger,
  title={YingMusic-Singer: Zero-shot Singing Voice Synthesis and Editing with Annotation-free Melody Guidance},
  author={Zheng, Junjie and Hao, Chunbo and Ma, Guobin and Zhang, Xiaoyu and Chen, Gongyu and Ding, Chaofan and Chen, Zihao and Xie, Lei},
  journal={arXiv preprint arXiv:2512.04779},
  year={2025}
}
```

---

## License 📝

This project is released under the **MIT License**.
