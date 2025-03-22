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
from dataset import VideoSet, ValSet
from model_structure import Model
from utils import Tokenlizer, OverLoss


'''
For testing the model.
'''

if __name__ == "__main__":
    device = torch.device('cuda')
    val_set = ValSet('../database/test/test') # set the path for test
    valloader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    model = Model().float().to(device)
    tokenizer = Tokenlizer()


    epoch = 10
    # load sate_dict
    state_dicts = torch.load('./modelcache/epoch%d.pth' % epoch) # set the state dict path
    model.load_state_dict(state_dicts['state_dict'])
    del state_dicts
    torch.cuda.empty_cache()

    # conduct 5 random tests. Note the final quality prediction for the video was obtained by averaging the results of these
    # 5 runs.
    # You shuold run make_output.py to get final averaged result.
    model.eval()
    for r in range(5):
        results = {}
        with torch.no_grad():
            for i, (frame_384, frame_256, frame_256_f, frames, prmt, ns) in enumerate(tqdm(valloader)):
                frame_384 = frame_384.to(torch.float).to(device).squeeze(1)
                frame_256 = frame_256.to(torch.float).to(device).squeeze(1)
                frame_256_f = frame_256_f.to(torch.float).to(device).squeeze(1)
                frames = frames.transpose(1, 2).to(torch.float).to(device).squeeze(1)
                ids_blip = tokenizer.blip(prmt).to(device)
                ids_clip = tokenizer.clip(prmt).to(device)
                qp, cls_p, qp_mix = model(frame_384, frame_256, frame_256_f, frames, ids_blip, ids_clip)
                qp_mix = qp_mix.view(-1).to('cpu').numpy()
                ns = ns.view(-1).numpy().astype(int)
                for i in range(qp_mix.shape[0]):
                    results[ns[i]] = qp_mix[i]
        with open('./results/results/ep%d_%d.pkl' % (epoch, r), 'wb') as f:
            pickle.dump(results, f)

