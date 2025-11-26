# YingMusic-Singer: Zero-shot Singing Voice Synthesis and Editing with Annotation-free Melody Guidance

---

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-YingMusic--Singer-blue)](tech_report/YingMusic-Singer_tech_report.pdf)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-YingMusic--Singer-yellow)](https://huggingface.co/GiantAILab/YingMusic-Singer)

</div>

---

## Overview ✨

<p align="center">
  <img src="figs/head.jpeg" width="720" alt="pipeline">
</p>

**YingMusic-Singer** is a unified framework for **Zero-shot Singing Voice Synthesis (SVS) and Editing**, driven by **Annotation-free Melody Guidance**. Addressing the scalability challenges of real-world applications, our system eliminates the reliance on costly phoneme-level alignment and manual melody annotations. It enables **arbitrary lyrics to be synthesized or edited with any reference melody** in a zero-shot manner.

Our approach leverages a **Diffusion Transformer (DiT)** based generative model, incorporating a pre-trained melody extraction module to derive MIDI information directly from reference audio. By introducing a structured guidance mechanism and employing **Flow-GRPO reinforcement learning**, we achieve superior pronunciation clarity, melodic accuracy, and musicality without requiring fine-grained alignment.

### 🔧 Key Features

- **Unified Synthesis & Editing**: Supports both zero-shot singing voice synthesis and precise singing voice editing within a single framework.
- **Annotation-free Melody Guidance**: Derives melody directly from reference audio, removing the need for manual MIDI or phoneme-level annotations.
- **Zero-Shot Capabilities**: Synthesize or edit high-quality singing voices from arbitrary lyrics and reference melodies without training on the target voice.
- **Flow-GRPO Reinforcement Learning**: Optimizes pronunciation, melodic accuracy, and musicality via multi-objective rewards.
- **Structured Guidance Mechanism**: Enhances melodic stability and coherence using similarity distribution constraints.
- **Robust Generalization**: Outperforms existing methods in zero-shot synthesis and lyric replacement scenarios.

---

<p align="center">
  <img src="resources/imgs/svs_main.svg" width="720" alt="pipeline">
</p>

---

## News & Updates 🗞️

- **2025-11-26**: Released the beta version's inference code and model checkpoints.
- **2025-11-26**: Released the technical report.

---

## Installation 🛠️

```bash
git clone https://github.com/GiantAILab/YingMusic-Singer.git
cd YingMusic-Singer

conda create -n singer python=3.10
conda activate singer
pip install -r requirements.txt
```

---

## Quick Start 🚀

### 1. Singing Voice Synthesis

```bash
cd accom_separation
bash infer.sh
```

### 2. Singing Voice Editing

```bash
bash my_infer.sh
```

---

## Development Roadmap & TODO 🗺️

- [x] Release beta version inference code and model checkpoints (currently supports Chinese & lower audio quality).
- [ ] Release V1 Version: Support for Chinese & English singing with higher audio quality and better generalization.
- [ ] Release training code.

---

## Acknowledgements 🙏

We would like to express our gratitude to the following projects for their contributions:

- **[F5-TTS](https://github.com/SWivid/F5-TTS)**: For model parameter initialization and key features in the DiT decoder.
- **[SOME](https://github.com/openvpi/SOME)**: For the Singing-Oriented MIDI Extractor used as a melody teacher in our online melody learning.
- **[Vocos](https://huggingface.co/charactr/vocos-mel-24khz)**: For the high-quality vocoder.
- **[DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)**: For providing valuable ideas and inspiration.

## License 📝

This project is released under the **MIT License**.
