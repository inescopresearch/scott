import torch

class TopKAccuracy():

    def __init__(self, k):
        self.k = k
        self.num_correct = 0
        self.total_samples = 0

    def update_metric(self, logits, targets):
        _, topk_pred = torch.topk(logits, k=self.k, dim=1)
        self.num_correct += (topk_pred == targets.view(-1,1)).sum().item()
        self.total_samples += len(logits)
    
    def get_value(self):
        return 100 * self.num_correct / self.total_samples
    
class MeanPerClassAccuracy():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.correct_per_class = torch.zeros(num_classes, dtype=torch.int64)
        self.samples_per_class = torch.zeros(num_classes, dtype=torch.int64)

    def update_metric(self, logits, targets):
        _, top1_pred = torch.topk(logits, k=1, dim=1)
        top1_pred = top1_pred.view(-1)
        
        for i in range(self.num_classes):
            class_mask = (targets == i)
            self.correct_per_class[i] += (top1_pred[class_mask] == targets[class_mask]).sum().item()
            self.samples_per_class[i] += class_mask.sum().item()
    
    def get_value(self):
        valid_classes = self.samples_per_class > 0
        per_class_accuracy = torch.zeros(self.num_classes, dtype=torch.float32)
        per_class_accuracy[valid_classes] = self.correct_per_class[valid_classes].float() / self.samples_per_class[valid_classes].float()
        return per_class_accuracy.mean().item() * 100

    def get_top1_per_class(self):
        valid_classes = self.samples_per_class > 0
        per_class_accuracy = torch.ones(self.num_classes, dtype=torch.float32) * -1
        per_class_accuracy[valid_classes] = self.correct_per_class[valid_classes].float() / self.samples_per_class[valid_classes].float()
        return dict([(idx, top1.item()) for idx, top1 in enumerate(per_class_accuracy)])

class LossMetric():

    def __init__(self):
        self.value = 0
        self.total_samples = 0

    def update_metric(self, value, num_samples):
        self.value += value * num_samples
        self.total_samples += num_samples

    def get_value(self):
        return self.value  / self.total_samples