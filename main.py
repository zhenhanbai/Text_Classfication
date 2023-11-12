import argparse
from models.Bert import Bert_model
from datasets.data_loader import data_load
from utils.util import training_model
from utils.metrics import evaluate
import os
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='bert-base-uncased', type=str, help="Pretrained BERT model name")
parser.add_argument("--model_path", default='/Users/zhenhan/Desktop/深度学习/NLP_Project/model_hub/bert', type=str, help="Pretrained BERT model path")
parser.add_argument("--dataset_dir", default='/Users/zhenhan/Desktop/深度学习/data/aclImdb.csv', type=str, help="Dataset location")
parser.add_argument("--model_dir", default='/Users/zhenhan/Desktop/深度学习/NLP_Project/checkpoints', type=str, help="Save model location")
parser.add_argument("--num_classes", default=2, type=int, help="Number of classes (binary classification)")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
parser.add_argument("--max_seq_len", default=256, type=int, help="Maximum sequence length")
parser.add_argument("--num_epochs", default=2, type=int, help="Number of training epochs")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate")
parser.add_argument("--device", default='mps', type=str, help="Device for training (cuda or cpu)")
args = parser.parse_args()

"""
python train.py \
    --dataset_dir "data" \
    --device "gpu" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --epochs 100
"""

if __name__ == "__main__":

    model = Bert_model(args)
    train_loader, val_loader, test_loader = data_load(args)
    print(len(train_loader), len(val_loader), len(test_loader))

    his = training_model(args, model, train_loader, test_loader)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    acc = evaluate(model, test_loader, args)
    print(acc)
    print(his)