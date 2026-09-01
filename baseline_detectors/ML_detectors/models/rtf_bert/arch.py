import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    BERT + Classification Head (Paper Section 3.5.1)
    """

    def __init__(self, model_name="bert-base-uncased", num_classes=6, hidden_dim=768):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Figure4. Based on the graph.
        # Pooler_output = [CLS] token -> Linear -> Tanh
        cls_output = outputs.pooler_output

        x = self.dropout(cls_output)
        logits = self.fc(x)
        return logits
