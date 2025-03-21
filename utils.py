import open_clip
import torch
from torch import nn
from transformers import AutoProcessor


class LossGroupQP(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def rank_loss(self,y_pred, y):
        if y_pred.dim()==1:
            y_pred=y_pred.unsqueeze(-1)
        if y.dim()==1:
            y=y.unsqueeze(-1)
        ranking_loss = torch.nn.functional.relu(
            (y_pred - y_pred.t()) * torch.sign((y.t() - y))
        )
        scale = 1 + torch.max(ranking_loss)
        return (
                torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
        ).float()

    def plcc_loss(self,y_pred, y):
        y_pred=y_pred.view(-1)
        y=y.view(-1)
        sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
        y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
        sigma, m = torch.std_mean(y, unbiased=False)
        y = (y - m) / (sigma + 1e-8)
        loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
        rho = torch.mean(y_pred * y)
        loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
        return ((loss0 + loss1) / 2).float()
    def forward(self,y_pred,y):
        return self.plcc_loss(y_pred,y)+0.3*self.rank_loss(y_pred,y)

class Tokenlizer:
    def __init__(self):
        ckpt = "google/siglip2-base-patch16-512"
        self.blip_ = AutoProcessor.from_pretrained(ckpt)
        self.clip_ = open_clip.get_tokenizer('convnext_base_w')

    def blip(self,text):
        return self.blip_(text=text, padding=True,truncation=True, return_tensors="pt",max_length=64).data['input_ids'][:,0:64]

    def clip(self,text):
        return self.clip_(text)

class OverLoss(nn.modules.loss._Loss):
    def __init__(self,w=[1,0.1]):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.rank_loss = LossGroupQP()
        self.w_single = w[0]
        self.w_mix = w[1]
    def forward(self,qp,cls_p,qp_mix,q,cls):
        loss=0
        for k, v in qp.items():
            loss+=self.w_single *self.rank_loss(v,q)
        for k,v in cls_p.items():
            loss+=self.w_single * self.ce(v,cls)*0.1
        loss+=self.w_mix*self.rank_loss(qp_mix,q)
        return loss

    def set_weights(self,w_single=1,w_mix=0.1):
        self.w_single=w_single
        self.w_mix=w_mix
