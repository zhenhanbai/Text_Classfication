from transformers import BertModel, BertTokenizer
import torch.nn as nn

class Bert_model(nn.Module):
    def __init__(self, args):
        super(Bert_model, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(768, args.num_classes)
        print("Initialized: Bert_model")

    """
    取出隐藏层的后四层
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    pooled = torch.sum(torch.stack(hidden_states[-4:]), dim=0)
    """
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]
        out = self.fc(pooled)
        return out
