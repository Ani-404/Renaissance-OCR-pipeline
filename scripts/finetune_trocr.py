from __future__ import annotations

"""
Fine-tune TrOCR on image-text pairs.
Expected CSV columns: image_path,text
"""

import argparse
from dataclasses import dataclass

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel


@dataclass
class OCRItem:
    image_path: str
    text: str


class OCRDataset(Dataset):
    def __init__(self, csv_path: str, processor: TrOCRProcessor, max_target_length: int = 256):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for printed OCR")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    parser.add_argument("--out", default="models/trocr-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)

    train_ds = OCRDataset(args.train_csv, processor)
    eval_ds = OCRDataset(args.eval_csv, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        num_train_epochs=args.epochs,
        fp16=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
    )

    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)


if __name__ == "__main__":
    main()
