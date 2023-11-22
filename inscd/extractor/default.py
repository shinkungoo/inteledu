import torch
import torch.nn as nn

from .._base import _Extractor


class Default(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, device, dtype):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num

        self.device = device
        self.dtype = dtype

        self.__student_emb = nn.Embedding(self.student_num, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.__diff_emb = nn.Embedding(self.exercise_num, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)

        self.__emb_map = {
            "mastery": self.__student_emb,
            "diff": self.__diff_emb,
            "disc": self.__disc_emb,
            "knowledge": self.__knowledge_emb
        }
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def extract(self, student_id, exercise_id, q_mask):
        student_ts = self.__student_emb(student_id)
        diff_ts = self.__diff_emb(exercise_id)
        disc_ts = self.__disc_emb(exercise_id)
        knowledge_ts = self.__knowledge_emb.weight
        return student_ts, diff_ts, disc_ts, knowledge_ts

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can detach {} from embeddings.".format(self.__emb_map.keys()))
        self.__emb_map["mastery"] = self.__student_emb
        self.__emb_map["diff"] = self.__diff_emb
        self.__emb_map["disc"] = self.__disc_emb
        self.__emb_map["knowledge"] = self.__knowledge_emb
        return self.__emb_map[item].weight

