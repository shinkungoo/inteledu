import torch
import torch.nn as nn
import torch.optim as optim

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import CDMFKC_IF
from ...extractor import CDMFKC_EX


class CDMFKC(_CognitiveDiagnosisModel):
    def __init__(self):
        """
        Description:
        Cognitive Diagnosis Model Focusing on Knowledge Concepts (CDMFKC)
        Sheng Li et al. Cognitive Diagnosis Focusing on Knowledge Concepts. CIKM'22
        """
        super().__init__()

    def build(self, datahub, if_type='cdmfkc', g_impact_a=0.5, g_impact_b=0.5, hidden_dims: list = None, dropout=0.5,
              device="cpu",
              dtype=torch.float32, **kwargs):
        self.student_num = datahub.student_num
        self.exercise_num = datahub.exercise_num
        self.knowledge_num = datahub.knowledge_num

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.extractor = CDMFKC_EX(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            device=device,
            dtype=dtype
        )
        if if_type == 'cdmfkc':
            self.inter_func = CDMFKC_IF(
                g_impact_a=g_impact_a,
                g_impact_b=g_impact_b,
                knowledge_num=self.knowledge_num,
                hidden_dims=hidden_dims,
                dropout=dropout,
                device=device,
                dtype=dtype)

    def train(self, datahub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=2e-3, weight_decay=0.0005, batch_size=256):
        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa", 'ap']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr, "weight_decay": weight_decay},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr, "weight_decay": weight_decay}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self._train(datahub=datahub, set_type=set_type,
                        valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                        batch_size=batch_size, loss_func=loss_func, optimizer=optimizer)

    def predict(self, datahub, set_type, batch_size=256, **kwargs):
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
        if metrics is None:
            metrics = ["acc", "auc", "f1", "doa", 'ap']
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics, batch_size=batch_size)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def load(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.extractor.load_state_dict(torch.load(ex_path))
        self.inter_func.load_state_dict(torch.load(if_path))

    def save(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        torch.save(self.extractor.state_dict(), ex_path)
        torch.save(self.inter_func.state_dict(), if_path)

    def get_attribute(self, attribute_name):
        if attribute_name == 'mastery':
            return self.diagnose().detach().cpu().numpy()
        elif attribute_name == 'diff':
            return self.inter_func.transform(self.extractor["diff"],
                                             self.extractor["knowledge"]).detach().cpu().numpy()
        elif attribute_name == 'knowledge':
            return self.extractor["knowledge"].detach().cpu().numpy()
        else:
            return None
