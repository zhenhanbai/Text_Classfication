import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_seq_len):
        self.texts = text  # List of texts
        self.labels = label # List of labels (0 or 1)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]


        inputs = self.tokenizer(
            text, truncation=True, 
            return_tensors='pt', 
            padding='max_length',  # Update here
            max_length=self.max_seq_len)


        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load the data and split into train and test sets
def data_load(args):
    data = pd.read_csv(args.dataset_dir)
    texts = data['text']
    labels = data['label']
    train_texts, tmp_texts, train_labels, tmp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(tmp_texts, tmp_labels, test_size=0.33, random_state=42)

    train_texts = train_texts.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    test_texts = test_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)
    test_labels = test_labels.reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # Create data loaders
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(train_loader)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, args.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, args.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
