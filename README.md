# 🇷🇺 Russian Speech Recognition: Benchmarking Architectures on the Golos Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-Golos-green)](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd)

## 📖 Project Overview
This project presents a comparative analysis of Automatic Speech Recognition (ASR) architectures evaluated on the **SberDevices Golos** dataset (Russian). The study investigates the evolution of ASR technology by benchmarking three distinct approaches:
1.  **Baseline:** Training a traditional CRNN (DeepSpeech 2) from scratch.
2.  **Experiment:** Cross-lingual transfer learning (HuBERT).
3.  **SOTA:** Domain adaptation of Transformers (Wav2Vec 2.0) and Zero-Shot evaluation (OpenAI Whisper).

We demonstrate that **Domain Adaptation** is the superior strategy for specific acoustic environments, achieving **8.17% WER** and outperforming larger generalized models.

---

## 📊 The Dataset: SberDevices Golos
We utilized the **Golos** dataset (Crowd Split), a challenging corpus representative of real-world acoustic environments provided by SberDevices.

| Feature | Description |
| :--- | :--- |
| **Source** | Crowdsourced recordings via mobile phones. |
| **Domain** | Voice assistant commands, search queries, slang, and brand names. |
| **Complexity** | **Unconstrained Acoustics:** Includes background noise, echo, and varying microphone quality.<br>**Short Utterances:** Lack of semantic context (commands vs. sentences). |
| **Size** | Total: ~1,240 hours. <br>**Used:** ~10 hours (Crowd Split) for training/benchmarking. |
| **Preprocessing** | Resampled to 16kHz; Text normalized to lowercase Cyrillic (punctuation removed). |

---

## 🛠️ Models & Architectures

We implemented and evaluated the following architectures to trace the progression of ASR technology:

### 1. DeepSpeech 2 (Baseline)
* **Architecture:** CRNN (Convolutional Recurrent Neural Network).
* **Method:** Trained from scratch using CTC Loss.
* **Goal:** To establish a baseline and demonstrate the high data requirements of RNN-based models.

### 2. HuBERT (Cross-Lingual Experiment)
* **Architecture:** Transformer (Hidden Unit BERT).
* **Method:** Pre-trained on **English** (LibriSpeech) $\to$ Fine-tuned on **Russian**.
* **Goal:** To test the transferability of acoustic features across languages.

### 3. Wav2Vec 2.0 (The Specialist)
* **Architecture:** Transformer (Self-Supervised).
* **Method:** **Domain Adaptation**. Utilized a model pre-trained on generic Russian audio (`wav2vec2-large-xlsr-53-russian`) and fine-tuned specifically on the Golos command dataset.
* **Goal:** To combine general linguistic knowledge with domain-specific vocabulary.

### 4. OpenAI Whisper (The Generalist)
* **Architecture:** Encoder-Decoder Transformer (Weak Supervision).
* **Method:** Zero-Shot Inference (Small).
* **Goal:** To benchmark against the industry State-of-the-Art and analyze generalization capabilities.

---

## 🏆 Comparative Results

Models were evaluated using **Word Error Rate (WER)** and **Character Error Rate (CER)** on the unseen test split.

| Model | Architecture | Training Method | WER 📉 | CER 📉 | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSpeech 2** | CRNN | From Scratch | **95.62%** | **53.42%** | **Failed.** High acoustic confusion; insufficient training data for convergence. |
| **HuBERT (En)** | Transformer | Cross-Lingual | **57.11%** | **15.02%** | **Promising.** Learned acoustics but lacked Russian grammar knowledge. |
| **Whisper** | Transformer | Zero-Shot | **34.86%*** | **18.86%** | **Formatting Mismatch.** High semantic accuracy, but penalized for outputting formatted text (e.g., "60" vs "sixty"). |
| **Wav2Vec 2.0** | Transformer | **Domain Adaptation** | **8.17%** | **1.51%** | **Best Performance.** Perfect alignment of acoustics and domain vocabulary. |

### 🧪 Key Findings
1.  **Transfer Learning is Essential:** Training from scratch on <50 hours of data yields unusable results (95% WER), whereas Transfer Learning yields acceptable results even with cross-lingual models.
2.  **Domain Adaptation Beats Generalization:** While Whisper is "smarter" (understands context better), the Wav2Vec 2.0 model fine-tuned on the specific dataset achieved significantly lower error rates by learning the specific "slang" and formatting rules of the domain.
3.  **Generalization Capability:** In our Pangram stress test (out-of-domain vocabulary), the Wav2Vec 2.0 model successfully transcribed complex words (*"French rolls"*) that were not present in the training commands, proving robust acoustic modeling.

---


🔗 References
Dataset: SberDevices Golos

Wav2Vec 2.0: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", 2020.

DeepSpeech 2: Amodei et al., "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin", 2015.
