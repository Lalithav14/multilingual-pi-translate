# 🌐 Multilingual Pi Translate

> **Efficient Multilingual Machine Translation using LoRA-fine-tuned mBART**

This project fine-tunes the `facebook/mbart-large-50-many-to-many-mmt` model using **LoRA (Low-Rank Adaptation)** via HuggingFace PEFT, enabling low-resource, parameter-efficient training of a multilingual machine translation system.

Supports live translation between 6+ languages via a **Gradio Web App**, and deployable on **Hugging Face Spaces**.

---

## 🚀 Features

- ✅ Multilingual machine translation (English, French, Hindi, German, Spanish, Chinese)
- ✅ LoRA parameter-efficient fine-tuning (PEFT)
- ✅ Based on `facebook/mbart-large-50-many-to-many-mmt`
- ✅ Trainable on free-tier GPUs (Colab, Kaggle)
- ✅ Interactive Gradio UI (local + web)
- ✅ Deployable to HuggingFace Spaces

---

## Languages Supported (currently)

| Language | Token |
|----------|-------|
| English  | `en_XX` |
| French   | `fr_XX` |
| German   | `de_DE` |
| Hindi    | `hi_IN` |
| Spanish  | `es_XX` |
| Chinese  | `zh_CN` |

Add more by modifying `language_codes` in `app.py`.

---

## Setup

```bash
git clone https://github.com/Lalithav14/multilingual-pi-translate.git
cd multilingual-pi-translate
pip install -r requirements.txt

