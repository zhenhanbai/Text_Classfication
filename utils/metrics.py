import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricReport():
    """
    Evaluation metrics for binary classification task.
    """

    def __init__(self, name='MetricReport'):
        super(MetricReport, self).__init__()
        self._name = name
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.y_prob = None
        self.y_true = None

    def update(self, probs, labels):
        """
        Update the probability and label
        """
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs.numpy(), axis=0)
        else:
            self.y_prob = probs.numpy()
        if self.y_true is not None:
            self.y_true = np.append(self.y_true, labels.numpy(), axis=0)
        else:
            self.y_true = labels.numpy()

    def accumulate(self):
        """
        Returns accuracy, precision, recall, and F1 score
        """
        y_pred = (self.y_prob > 0.5).astype(int)  # 设置阈值为0.5，得到预测的二分类标签
        accuracy = accuracy_score(y_true=self.y_true, y_pred=y_pred)
        precision = precision_score(y_true=self.y_true, y_pred=y_pred)
        recall = recall_score(y_true=self.y_true, y_pred=y_pred)
        f1 = f1_score(y_true=self.y_true, y_pred=y_pred)
        return accuracy, precision, recall, f1

    def name(self):
        """
        Returns metric name
        """
        return self._name

def evaluate(model, test_loader, args):
        model.eval()
        val_metric_sum = 0.0

        with torch.no_grad():
            for val_step, batch in enumerate(test_loader, 1):
                batch = {k: v.to(args.device) for k, v in batch.items()}  # 将数据移动到GPU设备上

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                logits = model(input_ids, attention_mask)

                _, predictions = torch.max(logits, dim=1)
                correct = torch.sum(predictions == labels)
                val_metric = correct.item() / labels.size(0)

                val_metric_sum += val_metric
            
            acc = val_metric_sum / val_step
        return acc