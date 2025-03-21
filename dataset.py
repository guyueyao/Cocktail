import decord
decord.bridge.set_bridge('torch')
from decord import cpu,VideoReader
import torchvision.transforms.v2.functional as VF
from torchvision.transforms.v2 import RandomCrop
import numpy as np
import json
import os
import pickle
import random
from torch.utils.data import Dataset,DataLoader
class VideoSet(Dataset):  # 继承Dataset
    def __init__(self,splits,root_path,test=False):  # __init__是初始化该类的一些基础参数
        images = np.arange(1, 23821)
        scores,prompts = self.get_scores()
        self.images = images[splits]
        self.scores,self.prompts = scores[splits],prompts[splits]
        with open('./codesdict.pkl', 'rb') as f:
            d = pickle.load(f)
        cls=d['maps']
        self.cls=cls[splits]
        self.test=test
        self.Rcrop256=RandomCrop(256,pad_if_needed=True)
        self.Rcrop384=RandomCrop(384,pad_if_needed=True)
        self.root_path=root_path

    def get_scores(self):
        with open('train.json', 'r') as file:
            data = json.load(file)
        scores = np.zeros((23820,))
        prompts=[]
        for i in range(23820):
            scores[i] = data[i]['gt_score']
            prompts.append(data[i]['prompt'])
        return scores,np.array(prompts)
    def resize_imgs(self,img,size):
        t,h,w,c=img.shape
        assert c==3
        img=img.permute(0,3,1,2)
        min_s=min(h,w)
        h_=int(h/min_s*size)
        w_=int(w/min_s*size)
        img=VF.resize(img, size=[h_, w_])
        img=VF.center_crop(img, size)
        return img
    def crop_imgs(self,img):
        img = img.permute(0, 3, 1, 2)
        t,c,h, w, = img.shape
        assert c == 3
        img=self.Rcrop384(img)
        return img

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        n=self.images[index]
        with open(os.path.join(self.root_path,'%d.mp4'%n), 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            T=len(vr)
            id = random.randint(0, T - 9)
            frames = vr[id:id+8]
            # frames=vr.get_batch(ids)
        id = random.randint(0, 7)
        frame_384=self.resize_imgs(frames[id:id+1],384)
        id = random.randint(0, 7)
        frame_256 = self.resize_imgs(frames[id:id + 1], 256)
        frames=self.crop_imgs(frames)
        id = random.randint(0, 7)
        frame_256_f=VF.center_crop(frames[id:id+1],256)
        score=self.scores[index]
        cls=self.cls[index]
        prmt=self.prompts[index]
        return frame_384,frame_256,frame_256_f,frames,score,cls,prmt  # 返回该样本

class ValSet(Dataset):  # 继承Dataset
    def __init__(self,root_path):  # __init__是初始化该类的一些基础参数
        self.images = np.arange(27223,34029+1)
        self.root_path=root_path
        self.prompts=[]
        with open('test_no_mos.json', 'r') as file:
            data = json.load(file)
        for i in range(6807):
            self.prompts.append(data[i]['prompt'])

    def resize_imgs(self, img, size):
        t, h, w, c = img.shape
        assert c == 3
        img = img.permute(0, 3, 1, 2)
        min_s = min(h, w)
        h_ = int(h / min_s * size)
        w_ = int(w / min_s * size)
        img = VF.resize(img, size=[h_, w_])
        img = VF.center_crop(img, size)
        return img

    def crop_imgs(self, img):
        img = img.permute(0, 3, 1, 2)
        t, c, h, w, = img.shape
        assert c == 3
        img = VF.center_crop(img,384)
        return img
    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        n=self.images[index]
        with open(os.path.join(self.root_path,'%d.mp4'%n), 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            T=len(vr)
            id = random.randint(0, T - 9)
            frames = vr[id:id+8]
            # frames=vr.get_batch(ids)
        id = random.randint(0, 7)
        frame_384=self.resize_imgs(frames[id:id+1],384)
        id = random.randint(0, 7)
        frame_256 = self.resize_imgs(frames[id:id + 1], 256)
        frames=self.crop_imgs(frames)
        id = random.randint(0, 7)
        frame_256_f=VF.center_crop(frames[id:id+1],256)
        prmt=self.prompts[index]
        return frame_384,frame_256,frame_256_f,frames,prmt,n # 返回该样本