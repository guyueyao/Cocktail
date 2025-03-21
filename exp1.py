import os
import pickle
os.environ['http_proxy'] = '10.108.10.23:7890'
os.environ['https_proxy'] = '10.108.10.23:7890'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

import decord
decord.bridge.set_bridge('torch')
from torch.utils.data import Dataset, DataLoader
from dataset import VideoSet,ValSet
from model_structure import Model
from utils import Tokenlizer, OverLoss

if __name__ == "__main__":
    device = torch.device('cuda')
    indexs = np.random.permutation(23820)
    train_set = VideoSet(indexs,'/home1/Databases/AIGC-C/train')
    val_set = ValSet('/home1/Databases/AIGC-C/test')
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    valloader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    loss_qp = OverLoss()
    model= Model().float().to(device)
    tokenizer=Tokenlizer()

    optimer_head=AdamW([{'params': model.adapter.parameters(),},{'params': model.multi_head.parameters(),},{'params': model.mix_head.parameters(),}],
                       lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
    optimer = AdamW(model.parameters(),
                         lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01)
    for epoch in range(15):

        model.train()
        for i, (frame_384,frame_256,frame_256_f,frames,score,cls,prmt) in enumerate(tqdm(train_loader)):
            frame_384 = frame_384.to(torch.float).to(device).squeeze(1)
            frame_256=frame_256.to(torch.float).to(device).squeeze(1)
            frame_256_f = frame_256_f.to(torch.float).to(device).squeeze(1)
            frames=frames.transpose(1, 2).to(torch.float).to(device).squeeze(1)
            score=score.to(torch.float).to(device).view(-1)
            cls=cls.to(torch.int64).to(device).view(-1)
            ids_blip = tokenizer.blip(prmt).to(device)
            ids_clip= tokenizer.clip(prmt).to(device)
            if epoch < 5:
                qp,cls_p,qp_mix = model.cold_start(frame_384,frame_256,frame_256_f,frames,ids_blip,ids_clip)
                loss = loss_qp(qp,cls_p,qp_mix,score,cls)
                loss.backward()
                optimer_head.step()
                optimer_head.zero_grad()
            else:
                qp, cls_p, qp_mix = model(frame_384, frame_256, frame_256_f, frames, ids_blip, ids_clip)
                loss = loss_qp(qp, cls_p, qp_mix, score, cls)
                loss.backward()
                optimer.step()
                optimer.zero_grad()
        if epoch==8:
            loss_qp.set_weights(0.1,1)

        # if epoch<-5:
        #     scheduler_cd.step()
        if epoch >= 5:
            torch.save({'state_dict': model.state_dict(), 'optim_dict':optimer.state_dict()},
                       './modelcache/epoch%d.pth' % (epoch))

            model.eval()
            results={}
            with torch.no_grad():
                for i, (frame_384,frame_256,frame_256_f,frames,prmt,ns) in enumerate(tqdm(valloader)):
                    frame_384 = frame_384.to(torch.float).to(device).squeeze(1)
                    frame_256 = frame_256.to(torch.float).to(device).squeeze(1)
                    frame_256_f = frame_256_f.to(torch.float).to(device).squeeze(1)
                    frames = frames.transpose(1, 2).to(torch.float).to(device).squeeze(1)
                    ids_blip = tokenizer.blip(prmt).to(device)
                    ids_clip = tokenizer.clip(prmt).to(device)
                    qp, cls_p, qp_mix = model(frame_384, frame_256, frame_256_f, frames, ids_blip, ids_clip)
                    qp_mix=qp_mix.view(-1).to('cpu').numpy()
                    ns=ns.view(-1).numpy().astype(int)
                    for i in range(qp_mix.shape[0]):
                        results[ns[i]]=qp_mix[i]
            with open('./results/ep%d_1.pkl'%epoch, 'wb') as f:
                pickle.dump(results, f)
            results = {}
            with torch.no_grad():
                for i, (frame_384, frame_256, frame_256_f, frames, prmt, ns) in enumerate(tqdm(valloader)):
                    frame_384 = frame_384.to(torch.float).to(device).squeeze(1)
                    frame_256 = frame_256.to(torch.float).to(device).squeeze(1)
                    frame_256_f = frame_256_f.to(torch.float).to(device).squeeze(1)
                    frames = frames.to(torch.float).to(device).squeeze(1)
                    ids_blip = tokenizer.blip(prmt).to(torch.float).to(device)
                    ids_clip = tokenizer.clip(prmt).to(torch.float).to(device)
                    qp, cls_p, qp_mix = model(frame_384, frame_256, frame_256_f, frames, ids_blip, ids_clip)
                    qp_mix = qp_mix.view(-1).to('cpu').numpy()
                    ns = ns.view(-1).numpy().astype(int)
                    for i in range(qp_mix.shape[0]):
                        results[ns[i]] = qp_mix[i]
            with open('./results/ep%d_2.pkl' % epoch, 'wb') as f:
                pickle.dump(results, f)
