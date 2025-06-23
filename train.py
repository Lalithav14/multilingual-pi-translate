from transformers import MBartForConditionalGeneration, MBartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Load multilingual dataset (English → French)
dataset = load_dataset("opus_books", "en-fr")

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBartTokenizer.from_pretrained(model_name)
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "fr_XX"

# Preprocessing
def preprocess_function(example):
    inputs = tokenizer(example["translation"]["en"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(example["translation"]["fr"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load base model
base_model = MBartForConditionalGeneration.from_pretrained(model_name)

# LoRA config for Seq2Seq
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, peft_config)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    num_train_epochs=2,
    save_strategy="epoch",
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(1000)),
    eval_dataset=tokenized_dataset["validation"].select(range(200)),
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
model.save_pretrained("multilingual-pi-translate")
tokenizer.save_pretrained("multilingual-pi-translate")


tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "hi_IN"  # Example: English to Hindi

def preprocess_function(example):
    inputs = tokenizer(example["translation"]["en"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(example["translation"]["hi"], truncation=True, padding="max_length", max_length=128)
    

