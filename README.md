# 🇷🇺 Russian Speech Recognition: Benchmarking Architectures

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)

## 📖 Project Overview
This project benchmarks the evolution of Automatic Speech Recognition (ASR) for the Russian language. We implemented and evaluated architectures ranging from manual mathematical decoders to custom CRNNs and state-of-the-art Transformers.

The study investigates:
1.  **Mathematical Fundamentals:** Implementing CTC decoding from scratch.
2.  **Legacy Architectures:** Training CRNN and DeepSpeech models to observe "data hunger."
3.  **Modern Transfer Learning:** Adapting HuBERT, Wav2Vec 2.0, and Whisper to Russian.

---

## 📊 Datasets
We utilized two distinct datasets to test different acoustic domains:

### 1. SberDevices Golos (Crowd Split)
* **Used for:** DeepSpeech, Wav2Vec 2.0, HuBERT, Whisper.
* **Domain:** Voice assistant commands (Short phrases, slang, brand names).
* **Environment:** Noisy, crowdsourced mobile audio.

### 2. Google Fleurs (ru_ru)
* **Used for:** CRNN (Experimental).
* **Domain:** Wikipedia sentences (Longer, formal speech).
* **Size:** 500 sentences cached for experimental training.

---

## 🛠️ Models & Architectures

We organized our experiments across three notebooks, implementing **6 distinct models**:

### 1. Rule-Based CTC (Manual Implementation)
* **Notebook:** `hubert + rule-based ctc.ipynb`
* **Type:** Mathematical Verification.
* **Method:** A manual implementation of the CTC Forward Algorithm and Greedy Search using raw PyTorch tensors.
* **Goal:** To verify the mathematical logic of alignment (Argmax → Collapse Repeats → Remove Blanks) independent of libraries.

### 2. HuBERT (Cross-Lingual)
* **Notebook:** `hubert + rule-based ctc.ipynb`
* **Type:** Transformer (Hidden Unit BERT).
* **Method:** **Cross-Lingual Transfer**. utilized a model pre-trained entirely on **English** (LibriSpeech) and fine-tuned on Russian.
* **Goal:** To prove that acoustic features (phonemes) are universal and transferrable between languages.

### 3. CRNN (Experimental)
* **Notebook:** `whisper + crnn + deepspeech.ipynb`
* **Architecture:** CNN (Feature Extractor) + BiLSTM (Sequence Modeling).
* **Training:** Trained for **250 Epochs** on the **Google Fleurs** dataset (Wikipedia data).
* **Goal:** To test a lightweight architecture on complex sentence data.
* **Outcome:** Despite 250 epochs, the model struggled to generalize on the diverse vocabulary of Wikipedia, resulting in high error rates.



### 4. DeepSpeech 2 (Baseline)
* **Notebook:** `whisper + crnn + deepspeech.ipynb`
* **Architecture:** DeepSpeech 2 (CNN + GRU + CTC).
* **Training:** Trained from scratch on **Golos** (Commands). Pushed for 40+ epochs until convergence (Loss < 0.5).
* **Goal:** To demonstrate the "memorization" capacity of RNNs.
* **Outcome:** The model successfully memorized the training data (Loss 0.43) but struggled to generalize to unseen test data, confirming the need for massive datasets for this architecture.



### 5. Wav2Vec 2.0 (The Specialist)
* **Notebook:** `wav2vec2 + small whisper.ipynb`
* **Architecture:** Transformer (Self-Supervised).
* **Method:** **Domain Adaptation**. utilized a model pre-trained on generic Russian audio and fine-tuned specifically on the Golos dataset.
* **Goal:** To achieve maximum accuracy by combining general linguistic knowledge with domain-specific vocabulary.



### 6. OpenAI Whisper (The Generalist)
* **Notebook:** `wav2vec2 + small whisper.ipynb`
* **Architecture:** Encoder-Decoder Transformer.
* **Method:** Zero-Shot Inference (Small).
* **Goal:** To benchmark against the industry State-of-the-Art.

---

## 🏆 Comparative Results

| Model | Notebook Source | WER 📉 | CER 📉 | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **CRNN (Fleurs)** | `whisper + crnn...` | **~0.82** | **~0.32** | **Insufficient Data.** 500 sentences were not enough for the LSTM to learn complex Wikipedia grammar. |
| **DeepSpeech (Golos)**| `whisper + crnn...` | **95.62%** | **53.42%** | **Overfitting.** Memorized training data (Loss 0.43) but failed on test data. |
| **HuBERT (En)** | `hubert + rule...` | **57.11%** | **15.02%** | **Proof of Concept.** Validated acoustic transfer, but lacked Russian grammar knowledge. |
| **Whisper (Small)** | `wav2vec2 + small...`| **34.86%** | **18.86%** | **Formatting Mismatch.** High semantic accuracy, penalized for outputting digits ("60") vs text. |
| **Wav2Vec 2.0** | `wav2vec2 + small...`| **8.17%** | **1.51%** | **The Winner.** Perfect alignment of acoustics and domain vocabulary. |

---

## 💻 Installation & Usage

### Requirements
```bash
pip install torch torchaudio transformers datasets jiwer soundfile librosa
