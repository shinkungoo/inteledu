import torch
import torch.nn as nn
from tqdm import tqdm

from .utility import init_interaction_function


class ComputeIF(nn.Module):
    def __init__(self,
                 student_number,
                 question_number,
                 knowledge_number):
        super(ComputeIF, self).__init__()
        self.student_emb = nn.Embedding(student_number, knowledge_number)
        self.difficulty = nn.Embedding(question_number, knowledge_number)
        self.discrimination = nn.Embedding(question_number, 1)

        # initialize
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_id, question, q_matrix_line, interaction_func):
        proficiency_level = torch.sigmoid(self.student_emb(student_id))
        difficulty = torch.sigmoid(self.difficulty(question))
        discrimination = torch.sigmoid(self.discrimination(question))

        input_x = interaction_func(discrimination, proficiency_level - difficulty, q_matrix_line)
        output = torch.sigmoid(input_x)

        return output.view(-1)


class Parameter:
    def __init__(self,
                 student_num: int,
                 question_num: int,
                 knowledge_num: int,
                 device: str):
        self.net = ComputeIF(student_num, question_num, knowledge_num)
        self.student_number = student_num
        self.question_number = question_num
        self.knowledge_number = knowledge_num
        self.interaction_function = init_interaction_function
        self.interaction_function_string = "initial interaction function"
        self.device = device

    def train(self, datahub, set_type, epoch, lr=0.002, init=True, **kwargs):
        # initialize
        if init:
            for name, param in self.net.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
        self.net = self.net.to(self.device)
        self.net.train()
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        dataloader = datahub.to_dataloader(
            batch_size=kwargs["batch_size"],
            dtype=torch.float32,
            set_type=set_type,
            label=True
        )
        with tqdm(total=epoch, desc="Training", unit="epoch") as pbar:
            for epoch_i in range(epoch):
                epoch_losses = []
                for batch_data in dataloader:
                    student_id, question, q_matrix_line, y = batch_data
                    student_id: torch.Tensor = student_id.to(self.device)
                    question: torch.Tensor = question.to(self.device)
                    q_matrix_line: torch.Tensor = q_matrix_line.to(self.device)
                    y: torch.Tensor = y.to(self.device)
                    pred: torch.Tensor = self.net(student_id,
                                                  question,
                                                  q_matrix_line,
                                                  self.interaction_function)
                    loss = loss_function(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.mean().item())
                pbar.update()

    def predict(self, datahub, set_type, batch_size, **kwargs):
        self.net = self.net.to(self.device)
        self.net.eval()
        dataloader = datahub.to_dataloader(
            batch_size=batch_size,
            dtype=torch.float32,
            set_type=set_type,
            label=True
        )
        y_pred = []
        for batch_data in tqdm(dataloader, "Evaluating"):
            student_id, question, q_matrix_line = batch_data
            student_id: torch.Tensor = student_id.to(self.device)
            question: torch.Tensor = question.to(self.device)
            q_matrix_line: torch.Tensor = q_matrix_line.to(self.device)
            pred: torch.Tensor = self.net(student_id,
                                          question,
                                          q_matrix_line,
                                          self.interaction_function)
            y_pred.extend(pred.detach().cpu().tolist())

        return y_pred

    def unpack(self):
        proficiency_level = self.net.student_emb(torch.arange(0, self.student_number).to()).detach().cpu().numpy()
        difficulty = self.net.difficulty(torch.arange(0, self.question_number).to()).detach().cpu().numpy()
        discrimination = self.net.discrimination(torch.arange(0, self.question_number).to()).detach().cpu().numpy()
        return proficiency_level, difficulty, discrimination,

    def update(self, interaction_func, interaction_func_str):
        self.interaction_function = interaction_func
        self.interaction_function_string = interaction_func_str
