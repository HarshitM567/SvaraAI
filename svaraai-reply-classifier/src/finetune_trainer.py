import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from src.data_processing import LABEL_MAP

def prepare_dataset(df):
    return Dataset.from_pandas(df)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./reply_classification_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='artifacts/hf_model')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=['label'])

    train = df.sample(frac=0.8, random_state=42)
    val = df.drop(train.index)
    ds = DatasetDict({
        'train': prepare_dataset(train),
        'validation': prepare_dataset(val)
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

    tokenized = ds.map(tokenize, batched=True)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABEL_MAP))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved fine-tuned model to", args.output_dir)

if __name__ == '__main__':
    main()
