import torch

class Bert_Config:
    def __init__(self):
        self.model_name = 'bert-base-uncased'  # Pretrained BERT model name
        self.num_classes = 2  # Binary classification, so 2 classes
        self.batch_size = 16
        self.max_seq_len = 128
        self.num_epochs = 3
        self.learning_rate = 2e-5
        self.hidden_size = 768
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')