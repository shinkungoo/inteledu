import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import NCD_IF, DP_IF, MIRT_IF, KANCD_IF, CDMFKC_IF, IRT_IF, KSCD_IF
from ...extractor import ORCDF_EX


class ORCDF(_CognitiveDiagnosisModel):
    def __init__(self):
        """
        Description:
        Oversmoothing-Resistant Cognitive Diagnosis Framework (ORCDF)
        Shuo Liu et al. ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems. KDD'24.
        """
        super().__init__()

    def build(self, datahub, latent_dim=32, device: str = "cpu", gcn_layers: int = 3, if_type='dp-linear'
              , keep_prob=0.9,
              dtype=torch.float64, hidden_dims: list = None, mode='all', flip_ratio=0.1, ssl_temp=0.8, ssl_weight=1e-2,
              **kwargs):
        self.student_num = datahub.student_num
        self.exercise_num = datahub.exercise_num
        self.knowledge_num = datahub.knowledge_num

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.device = device
        self.mode = mode
        self.flip_ratio = flip_ratio
        self.if_type = if_type

        # if 'tf' in self.mode:
        #     if if_type != 'kancd':
        #         latent_dim = self.knowledge_num
        # if if_type == 'kancd' or if_type == 'kscd' or if_type == 'mirt' or if_type == 'irt':
        #     latent_dim = 32
        # else:
        #     latent_dim = self.knowledge_num

        self.extractor = ORCDF_EX(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
            mode=mode,
            ssl_temp=ssl_temp,
            ssl_weight=ssl_weight
        )

        if if_type == 'ncd':
            self.inter_func = NCD_IF(knowledge_num=self.knowledge_num,
                                     hidden_dims=hidden_dims,
                                     dropout=0,
                                     device=device,
                                     dtype=dtype)
        elif 'dp' in if_type:
            self.inter_func = DP_IF(knowledge_num=self.knowledge_num,
                                    hidden_dims=hidden_dims,
                                    dropout=0,
                                    device=device,
                                    dtype=dtype,
                                    kernel=if_type)
        elif 'mirt' in if_type:
            self.inter_func = MIRT_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=32,
                device=device,
                dtype=dtype,
                utlize=True)
        elif 'kancd' in if_type:
            self.inter_func = KANCD_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype,
                hidden_dims=hidden_dims,
                dropout=0.5
            )
        elif 'cdmfkc' in if_type:
            self.inter_func = CDMFKC_IF(
                g_impact_a=0.5,
                g_impact_b=0.5,
                knowledge_num=self.knowledge_num,
                hidden_dims=hidden_dims,
                dropout=0.5,
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )
        elif 'irt' in if_type:
            self.inter_func = IRT_IF(
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )
        elif 'kscd' in if_type:
            self.inter_func = KSCD_IF(
                dropout=0.5,
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype)
        else:
            raise ValueError("Remain to be aligned....")

    def train(self, datahub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=5e-4, weight_decay=0.0005, batch_size=256):

        if self.mode == 'Q':
            ek_graph = np.zeros(shape=datahub.q_matrix.shape)
        else:
            ek_graph = datahub.q_matrix.copy()

        se_graph_right, se_graph_wrong = [self.__create_adj_se(datahub[set_type], is_subgraph=True)[i] for i in
                                          range(2)]
        se_graph = self.__create_adj_se(datahub[set_type], is_subgraph=False)

        if self.flip_ratio:
            def get_flip_data():
                np_response_flip = datahub[set_type].copy()
                column = np_response_flip[:, 2]
                probability = np.random.choice([True, False], size=column.shape,
                                               p=[self.flip_ratio, 1 - self.flip_ratio])
                column[probability] = 1 - column[probability]
                np_response_flip[:, 2] = column
                return np_response_flip

        graph_dict = {
            'right': self.__final_graph(se_graph_right, ek_graph),
            'wrong': self.__final_graph(se_graph_wrong, ek_graph),
            'response': datahub[set_type],
            'Q_Matrix': datahub.q_matrix.copy(),
            'flip_ratio': self.flip_ratio,
            'all': self.__final_graph(se_graph, ek_graph)
        }

        self.extractor.get_graph_dict(graph_dict)
        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa", 'ap']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr, "weight_decay": weight_decay},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr, "weight_decay": weight_decay}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self.extractor.get_flip_graph()
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

    def update_graph(self, data, q_matrix):
        se_graph_right, se_graph_wrong = [self.__create_adj_se(data, is_subgraph=True)[i] for i in
                                          range(2)]
        self.extractor.graph_dict['right'] = self.__final_graph(se_graph_right, q_matrix)
        self.extractor.graph_dict['wrong'] = self.__final_graph(se_graph_wrong, q_matrix)

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num)), np.zeros(
                    shape=(self.student_num, self.exercise_num))

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self.__get_csr(train_stu_right, train_exer_right,
                                          shape=(self.student_num, self.exercise_num))
            adj_se_wrong = self.__get_csr(train_stu_wrong, train_exer_wrong,
                                          shape=(self.student_num, self.exercise_num))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num))
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def __final_graph(self, se, ek):
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num: se_num] = se
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)

    @property
    def name(self):
        return 'orcdf'

