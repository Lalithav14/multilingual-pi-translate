import gradio as gr
from transformers import MBartForConditionalGeneration, MBartTokenizer
from peft import PeftModel

model_path = "multilingual-pi-translate"
base_model_name = "facebook/mbart-large-50-many-to-many-mmt"

base_model = MBartForConditionalGeneration.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = MBartTokenizer.from_pretrained(model_path)

# Supported language codes from mBART
language_codes = {
    "English": "en_XX",
    "French": "fr_XX",
    "German": "de_DE",
    "Hindi": "hi_IN",
    "Spanish": "es_XX",
    "Chinese": "zh_CN"
}

def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = language_codes[source_lang]
    tgt_lang_code = language_codes[target_lang]
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
        max_length=128
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

demo = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text"),
        gr.Dropdown(list(language_codes.keys()), value="English", label="Source Language"),
        gr.Dropdown(list(language_codes.keys()), value="French", label="Target Language"),
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="🌐 Multilingual Pi Translator",
    description="Translate between English, Hindi, French, German, Chinese, Spanish using a LoRA-fine-tuned mBART model."
)

if __name__ == "__main__":
    demo.launch()
