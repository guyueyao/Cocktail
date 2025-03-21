
import open_clip
from swin_backbone2 import swin_wrap
import torch
from transformers import AutoModel, AutoProcessor, AutoModelForImageClassification
import torch.nn as nn
import torch.nn.functional as F


class Blip2(nn.Module):
    def __init__(self):
        super(Blip2, self).__init__()
        self.device = torch.device("cuda")
        ckpt = "google/siglip2-base-patch16-384"
        model = AutoModel.from_pretrained(ckpt, device_map="auto")  # .eval()
        self.vision_model = model.vision_model.to(torch.float)
        self.text_model = model.text_model.to(torch.float)
        del model
        # self.visual_adapter = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 384), nn.GELU(), )
        # self.text_adapter = nn.Sequential(nn.Dropout(0.5), nn.Linear(768 * 2, 384), nn.GELU(), )
        # self.quality_predictor1 = nn.Linear(384, 1)
        # self.quality_predictor2 = nn.Linear(384, 1)
        # self.class_fi_head = nn.Linear(384, 22)

    @torch.no_grad()
    def __preprocess_siglip(self, images):
        if not images.shape[1] == 3:
            images = images.permute(0, 3, 1, 2)
        images.div_(255.0).sub_(0.5).div_(0.5)
        return images

    @torch.no_grad()
    def forward_text_features(self, input_ids):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        pooled_output = text_outputs[1]
        return pooled_output

    def forward_features(self, images):
        pixel_values = self.__preprocess_siglip(images)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            interpolate_pos_encoding=False,
        )
        pooled_output = vision_outputs[1]
        return pooled_output

    def forward(self, samples_f, texts_ids):
        # b,c,h,w=samples_c.shape
        feats_v = self.forward_features(samples_f)
        feats_txt = self.forward_text_features(texts_ids)
        # feats_v_a = self.visual_adapter(feats_v)
        b_v = feats_v.shape[0]
        b_txt = feats_txt.shape[0]
        if b_v != b_txt:
            feats_txt = feats_txt.unsqueeze(1).repeat(1, b_v // b_txt, 1).view(b_v, 768)
        return feats_v,feats_txt
        # feats_txt_a = self.text_adapter(torch.cat((feats_v, feats_txt), dim=-1))
        # qp1 = self.quality_predictor1(feats_v_a)
        # qp2 = self.quality_predictor2(feats_txt_a)
        # cls = self.class_fi_head(feats_v_a)
        # return qp1, qp2, cls

    def cold_start(self, samples_f, texts_ids):
        with torch.no_grad():
            feats_v = self.forward_features(samples_f)
            feats_txt = self.forward_text_features(texts_ids)
        feats_v_a = self.visual_adapter(feats_v)
        feats_txt_a = self.text_adapter(torch.cat((feats_v, feats_txt), dim=-1))
        qp1 = self.quality_predictor1(feats_v_a)
        qp2 = self.quality_predictor2(feats_txt_a)
        cls = self.class_fi_head(feats_v_a)
        return qp1, qp2, cls

class ConVNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        self.model = open_clip.create_model('convnext_base_w', pretrained='laion2B-s13B-b82K')

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],dtype=torch.float,device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.std =  torch.tensor([0.26862954, 0.26130258, 0.27577711],dtype=torch.float,device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # self.visual_adapter=nn.Sequential(nn.Dropout(0.5),nn.Linear(640,320),nn.GELU(),)
        # self.text_adapter=nn.Sequential(nn.Dropout(0.5),nn.Linear(640*2,320),nn.GELU(),)
        # self.quality_predictor1=nn.Linear(320,1)
        # self.quality_predictor2 = nn.Linear(320, 1)
        # self.class_fi_head=nn.Linear(320,22)
    @torch.no_grad()
    def __preprocess_clip(self, images):
        if not images.shape[1]==3:
            images = images.permute(0, 3, 1, 2)
        images.div_(255.0).sub_(self.mean).div_(self.std)
        return images

    @torch.no_grad()
    def forward_text_features(self,input_ids):
        return self.model.encode_text(input_ids,normalize=False)

    def forward_features(self, images): #256x256->768
        pixel_values = self.__preprocess_clip(images)
        vision_outputs = self.model.encode_image(pixel_values,normalize=False)
        return vision_outputs

    def forward(self,samples_f,texts_ids):
        # b,c,h,w=samples_c.shape
        feats_v=self.forward_features(samples_f)
        feats_txt=self.forward_text_features(texts_ids)
        # feats_v_a=self.visual_adapter(feats_v)
        b_v=feats_v.shape[0]
        b_txt=feats_txt.shape[0]
        if b_v!=b_txt:
            feats_txt=feats_txt.unsqueeze(1).repeat(1,b_v//b_txt,1).view(b_v,640)

        return feats_v,feats_txt
        # feats_txt_a=self.text_adapter(torch.cat((feats_v,feats_txt),dim=-1))
        # qp1=self.quality_predictor1(feats_v_a)
        # qp2=self.quality_predictor2(feats_txt_a)
        # cls=self.class_fi_head(feats_v_a)
        # return qp1,qp2,cls
    def cold_start(self,samples_f,texts_ids):
        with torch.no_grad():
            feats_v = self.forward_features(samples_f)
            feats_txt = self.forward_text_features(texts_ids)
        feats_v_a = self.visual_adapter(feats_v)
        feats_txt_a = self.text_adapter(torch.cat((feats_v, feats_txt), dim=-1))
        qp1 = self.quality_predictor1(feats_v_a)
        qp2 = self.quality_predictor2(feats_txt_a)
        cls = self.class_fi_head(feats_v_a)
        return qp1, qp2, cls

class Swinv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-small-patch4-window8-256")
        self.vision_model = model.swinv2.to(torch.float)
        del model
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float,device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.std =  torch.tensor([0.229, 0.224, 0.225],dtype=torch.float,device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # self.visual_adapter=nn.Sequential(nn.Dropout(0.5),nn.Linear(768,384),nn.GELU(),)
        # self.quality_predictor1=nn.Linear(384,1)
        # self.class_fi_head=nn.Linear(384,22)
    @torch.no_grad()
    def __preprocess_swinv2(self, images):
        if not images.shape[1]==3:
            images = images.permute(0, 3, 1, 2)
        images.div_(255.0).sub_(self.mean).div_(self.std)
        return images

    def forward_features(self, images): #256x256->768
        pixel_values = self.__preprocess_swinv2(images)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            interpolate_pos_encoding=False,
        )
        pooled_output = vision_outputs[1]
        return pooled_output

    def forward(self,samples_f):
        # b,c,h,w=samples_c.shape
        feats_v=self.forward_features(samples_f)
        return feats_v
        # feats_v_a=self.visual_adapter(feats_v)
        # qp1=self.quality_predictor1(feats_v_a)
        # cls=self.class_fi_head(feats_v_a)
        # return qp1,cls
    def cold_start(self,samples_f):
        with torch.no_grad():
            feats_v = self.forward_features(samples_f)
        feats_v_a = self.visual_adapter(feats_v)
        qp1 = self.quality_predictor1(feats_v_a)
        cls = self.class_fi_head(feats_v_a)
        return qp1, cls

class Swin3D(nn.Module):
    def __init__(self):
        super(Swin3D, self).__init__()
        self.device = torch.device('cuda')
        model = swin_wrap()
        ckeckpoint = torch.load(
            'swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb_20230612-8e082ff1.pth')
        model.load_state_dict(ckeckpoint['state_dict'], strict=False)
        self.visual_encoder = model.backbone
        self.visual_encoder.use_checkpoint = False
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1).to(self.device)
        self.std = torch.FloatTensor([58.395, 57.12, 57.375]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(
            self.device)
        # self.adapter = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 384), nn.GELU())
        # self.qp_head = nn.Linear(384, 1)
        # self.cls_head = nn.Linear(384, 22)

    def forward(self, x_c):
        x_c.sub_(self.mean).div_(self.std)
        f = self.visual_encoder(x_c)
        f = F.adaptive_avg_pool3d(f, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        return f
        # f = self.adapter(f)
        # qp = self.qp_head(f)
        # cls = self.cls_head(f)
        # return qp, cls

    def cold_start(self, x_c):
        with torch.no_grad():
            x_c.sub_(self.mean).div_(self.std)
            f = self.visual_encoder(x_c)
            f = F.adaptive_avg_pool3d(f, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        f = self.adapter(f)
        qp = self.qp_head(f)
        cls = self.cls_head(f)
        return qp, cls

class Adapter(nn.Module):
    def __init__(self,dims_in:dict={'blip_v':768,'blip_t':768*2,'conxt_v':640,'conxt_t':640*2,'swinv2_v':768,'swin3d_v':768},
        dims_out: dict = {'blip_v': 384, 'blip_t': 384, 'conxt_v': 340, 'conxt_t': 340, 'swinv2_v': 384, 'swin3d_v': 384},
                 ):
        super().__init__()
        self.adapter = nn.ModuleDict()
        for k,v in dims_in.items():
            self.adapter[k] = nn.Sequential(nn.Dropout(0.5), nn.Linear(v, dims_out[k]), nn.GELU())

    def forward(self,features:dict):
        features_a={}
        for k,v in features.items():
            if '_v' in k:
                features_a[k]=self.adapter[k](v)
            elif '_t' in k:
                features_a[k]=self.adapter[k](torch.cat((v,features[k.replace('_t','_v')]),dim=-1))
            else:
                raise ValueError(k)

        return features_a

class MultiHead(nn.Module):
    def __init__(self,
                 dims_in:dict={'blip_v':384,'blip_t':384,'conxt_v':340,'conxt_t':340,'swinv2_v':384,'swin3d_v':384},

                 ):
        super(MultiHead, self).__init__()
        self.QPs = nn.ModuleDict()
        self.CLSs = nn.ModuleDict()
        for k,v in dims_in.items():
            self.QPs[k] = nn.Linear(v,1)
            if '_v' in k:
                self.CLSs[k] = nn.Linear(v, 22)
    def forward(self,features:dict):
        QPs={}
        CLSs={}
        for k,v in features.items():
            QPs[k]=self.QPs[k](v)
            if '_v' in k:
                CLSs[k]=self.CLSs[k](v)
        return QPs,CLSs

class Mix_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Dropout(0.25),nn.Linear(384*4+680,384),nn.Dropout(0.2),nn.GELU(),nn.Linear(384,1))

    def forward(self,features:dict):
        features_list=[]
        for k,v in features.items():
            features_list.append(v)
        features=torch.cat(features_list,dim=-1)
        qp=self.head(features)
        return qp

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.blip=Blip2()
        self.conxt=ConVNext()
        self.swinv2=Swinv2()
        self.swin3d=Swin3D()
        self.adapter=Adapter()
        self.multi_head=MultiHead()
        self.mix_head=Mix_head()
        self.device = torch.device('cuda')

    def forward_features(self,frame_384,frame_256,frame_256_f,frames,blip_ids,clip_ids):
        blip_v,blip_t=self.blip(frame_384,blip_ids)
        conxt_v,conxt_t=self.conxt(frame_256,clip_ids)
        swinv2_v=self.swinv2(frame_256_f)
        swin3d_v=self.swin3d(frames)
        return {'blip_v':blip_v,'blip_t':blip_t,'conxt_v':conxt_v,'conxt_t':conxt_t,'swinv2_v':swinv2_v,'swin3d_v':swin3d_v}

    def forward(self,frame_384,frame_256,frame_256_f,frames,blip_ids,clip_ids):
        features=self.forward_features(frame_384,frame_256,frame_256_f,frames,blip_ids,clip_ids)
        features_a=self.adapter(features)
        qp,cls=self.multi_head(features_a)
        qp_mix=self.mix_head(features_a)
        return qp,cls,qp_mix

    def cold_start(self,frame_384,frame_256,frame_256_f,frames,blip_ids,clip_ids):
        with torch.no_grad():
            features=self.forward_features(frame_384,frame_256,frame_256_f,frames,blip_ids,clip_ids)
        features_a=self.adapter(features)
        qp,cls=self.multi_head(features_a)
        qp_mix=self.mix_head(features_a)
        return qp,cls,qp_mix
