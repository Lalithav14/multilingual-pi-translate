from transformers import MBartForConditionalGeneration, MBartTokenizer
from peft import PeftModel

model_path = "multilingual-pi-translate"
model_name = "facebook/mbart-large-50-many-to-many-mmt"

base_model = MBartForConditionalGeneration.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = MBartTokenizer.from_pretrained(model_path)

# Set source and target languages
tokenizer.src_lang = "en_XX"
tgt_lang = "fr_XX"
text = "I am building an AI project."

inputs = tokenizer(text, return_tensors="pt", padding=True)
generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print("Translation:", translation[0])
