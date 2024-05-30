from datasets import load_dataset
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from transformers import TrainingArguments, Trainer, default_data_collator
from datasets.features import ClassLabel
from transformers import TrainerCallback
from google.cloud import storage
from datasets import load_metric
from functools import partial
from PIL import Image
import numpy as np
import torch
import glob
import os


bucket_folder_dir = "Model2.0"
bucket_name="customtraining88"
local_path="Model3.0/checkpoint-500/"

# Initialize processor
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)


def upload_to_bucket(src_path, dest_bucket_name, dest_path):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(dest_bucket_name)
        if os.path.isfile(src_path):
            blob = bucket.blob(os.path.join(dest_path, os.path.basename(src_path)))
            blob.upload_from_filename(src_path)
            return
        for item in glob.glob(src_path + '/*'):
            if os.path.isfile(item):
                blob = bucket.blob(os.path.join(dest_path, os.path.basename(item)))
                blob.upload_from_filename(item)
            else:
                upload_to_bucket(item, dest_bucket_name, os.path.join(dest_path, os.path.basename(item)))


def load_custom_dataset():
    # Replace the path with your actual dataset path
    dataset = load_dataset('dataset.py')
    return dataset
    

def setup_custom_features(label_list):
    return {
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(ClassLabel(names=label_list)),
    }


def compute_metrics(p, label_list):
    metric = load_metric("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    # Load dataset
    dataset = load_custom_dataset()
    print('loading dataset completed')

    # Define label-related parameters
    features = dataset["train"].features
    label_column_name = "ner_tags"
    label_list = features[label_column_name].feature.names
    print('Define label-related parameters')

    features = dataset["train"].features
    column_names = dataset["train"].column_names
    image_column_name = "image_path"
    text_column_name = "words"
    boxes_column_name = "bboxes"
    label_column_name = "ner_tags"
    print('defining features done')

    from datasets.features import ClassLabel
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    else:
        label_list = get_label_list(dataset["train"][label_column_name])
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    num_labels = len(label_list)


    def prepare_examples(examples):
      images = [Image.open(path).convert("RGB") for path in examples['image_path']] #Image.open(examples[image_column_name])
      words = examples[text_column_name]
      boxes = examples[boxes_column_name]
      word_labels = examples[label_column_name]

      encoding = processor(images, words, boxes=boxes, word_labels=word_labels,truncation=True, padding="max_length", stride =128, max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)
      offset_mapping = encoding.pop('offset_mapping')
      overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
      return encoding


    from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

    
    # we need to define custom features for `set_format` (used later on) to work properly
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(ClassLabel(names=label_list)),
    })
    # Prepare datasets
    train_dataset = dataset["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    batch_size=32,) # specify the batch size)
    print('prepration of train dataset')

    # Prepare datasets
    eval_dataset = dataset["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    batch_size=32,) # specify the batch size)
    print('eval_dataset of train dataset')


    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    print('converting to torch')

    # Initialize LayoutLMv3ForTokenClassification model
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                             id2label=id2label,
                                                             label2id=label2id)
    print('model loaded')

    print('training arguments started')
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=local_dir,
        max_steps=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        no_cuda=False
    )
    print('training arguments finished')

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_list),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()
    
    # Call the upload function
    upload_to_bucket(local_path, bucket_name,bucket_folder_dir)
    
    
if __name__ == '__main__':
    main()
