import os
import pickle
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

'''
For traning the model.
'''

if __name__ == "__main__":
    device = torch.device('cuda')
    indexs = np.random.permutation(23820)
    train_set = VideoSet(indexs,'../database/train') # set the path for train
    val_set = ValSet('../database/test/test') # set the path for test
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=12, pin_memory=True,
                              persistent_workers=True)
    valloader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    loss_qp = OverLoss() # The PLCC and Rank loss from FAST-VQA is adopted.
    model= Model().float().to(device)
    tokenizer=Tokenlizer()

    optimer_head=AdamW([{'params': model.adapter.parameters(),},{'params': model.multi_head.parameters(),},{'params': model.mix_head.parameters(),}],
                       lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
    optimer = AdamW(model.parameters(),
                         lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01)
    # state_dicts=torch.load('./modelcache/epoch7.pth')
    # model.load_state_dict(state_dicts['state_dict'])
    # optimer.load_state_dict(state_dicts['optim_dict'])
    # del state_dicts
    # torch.cuda.empty_cache()
    for epoch in range(0,10):

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
            if epoch <5: # train adapters、quality network、FCs only for first 5 epoch.
                qp,cls_p,qp_mix = model.cold_start(frame_384,frame_256,frame_256_f,frames,ids_blip,ids_clip)
                loss = loss_qp(qp,cls_p,qp_mix,score,cls)
                loss.backward()
                optimer_head.step()
                optimer_head.zero_grad()
            else: # finetune entire model for last 5 epoch
                qp, cls_p, qp_mix = model(frame_384, frame_256, frame_256_f, frames, ids_blip, ids_clip)
                loss = loss_qp(qp, cls_p, qp_mix, score, cls)
                loss.backward()
                optimer.step()
                optimer.zero_grad()

        if  epoch>=7:
            loss_qp.set_weights(0.1, 1)
        # if epoch<-5:
        #     scheduler_cd.step()
        if epoch >= 5:
            torch.save({'state_dict': model.state_dict(), 'optim_dict':optimer.state_dict()},
                       './modelcache/epoch%d.pth' % (epoch))

            # conduct 5 random tests. Note the final quality prediction for the video was obtained by averaging the results of these
            # 5 runs.
            model.eval()
            for r in range(5):
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
                with open('./results/results/ep%d_%d.pkl'%(epoch,r), 'wb') as f:
                    pickle.dump(results, f)

