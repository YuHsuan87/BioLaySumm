from utils import *

from transformers import(
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from tqdm import tqdm
from datasets import Dataset


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

encoder_max_length = 6144
decoder_max_length = 512
batch_size = 1

def process_data_to_model_inputs(batch):
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

# construct training dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='PLOS')
parser.add_argument("--datatype", type=str, default="train")
args = parser.parse_args()
# training data
article_train, lay_sum_train, _ = load_task1_data(args)
train_dataset = {'article': article_train, 'abstract': lay_sum_train}
train_dataset = Dataset.from_dict(train_dataset)
# validation data
args.datatype = 'val'
article_val, lay_sum_val, _ = load_task1_data(args)
val_dataset = {'article': article_val, 'abstract': lay_sum_val}
val_dataset = Dataset.from_dict(val_dataset)

# --------------------test 300 nums of data-------------------
train_dataset = train_dataset.select(range(1000))
val_dataset = val_dataset.select(range(10))
# ------------------------------------------------------------


# map train data
train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched = True,
    batch_size = batch_size,
    remove_columns=["article", "abstract"]
)
# map val data
val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched = True,
    batch_size = batch_size,
    remove_columns=["article", "abstract"]
)

# the datasets should be converted into the PyTorch format
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

from rouge import Rouge
rouge = Rouge()
# the generation output, called pred.predictions as well as the gold label, called pred.label_ids.
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.get_scores(pred_str, label_str)[0]['rouge-2']

    return {
        "rouge2_precision": round(rouge_output['p'], 4),
        "rouge2_recall": round(rouge_output['r'], 4),
        "rouge2_fmeasure": round(rouge_output['f'], 4),
    }

led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)

# set generate hyperparameters
led.config.num_beams = 2
led.config.max_length = 512
led.config.min_length = 100
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3

# Training
model_name = 'long-2'
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    output_dir=f"./{model_name}",
    logging_steps=5,
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
)

trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()