import os.path as osp

import pytorch_lightning
import torch.nn.functional as F
from mmcv import Config
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from components import freeze_model, unfreeze_model, ImagePool, get_smooth_loss
from utils import EWMA
from .disp_net import DispNet
from .gan import GANLoss, NLayerDiscriminator
from .layers import SSIM, Backproject, Project
from .registry import MODELS
from .utils import *


import os
import mmcv
import numpy as np
import copy

import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
#import GroundedSegmentAnything.GroundingDINO.groundingdino.datasets.transforms as T
from GroundedSegmentAnything.GroundingDINO.groundingdino.datasets import transforms as T
from GroundedSegmentAnything.GroundingDINO.groundingdino.models import build_model
from GroundedSegmentAnything.GroundingDINO.groundingdino.util import box_ops
from GroundedSegmentAnything.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundedSegmentAnything.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

import cv2
import matplotlib.pyplot as plt
from torchvision import transforms


def build_disp_net(option, check_point_path):
    # create model
    model: pytorch_lightning.LightningModule = MODELS.build(name=option.model.name, option=option)
    model.load_state_dict(torch.load(check_point_path, map_location='cpu')['state_dict'])
    model.freeze()
    model.eval()

    # return
    return model

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def load_image(image_path):
        # load image
    image_pil = Image.fromarray(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


@MODELS.register_module(name='depsam')
class DepSAMModel(LightningModule):
    """
    The training process
    """
    def __init__(self, opt):
        super(DepSAMModel, self).__init__()

        #SAM cfg-----------
        self.config_file = "GroundedSegmentAnything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
        self.grounded_checkpoint = "GroundedSegmentAnything/groundingdino_swint_ogc.pth"  # change the path of the model
        self.sam_version = "vit_h"
        self.sam_checkpoint = "GroundedSegmentAnything/sam_vit_h_4b8939.pth"
        self.sam_hq_checkpoint = None
        self.use_sam_hq = "False"
        self.text_prompt = "light"
        self.box_threshold = 0.2
        self.text_threshold = 0.15
        self.device2 = "cuda"
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # initialize SAM
        #if self.use_sam_hq:
            #self.predictor = SamPredictor(sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint).to(self.device2))
        #else:
        with torch.no_grad(): 
            self.predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device2))
            self.sam_model = load_model(self.config_file, self.grounded_checkpoint, device=self.device2)

        



        self.opt = opt.model
        #opt_without_model = {k: v for k, v in opt.__dict__.items() if k in self.opt}
        #self.opt_seg = opt.split

        # components
        self.gan_loss = GANLoss('lsgan')
        self.image_pool = ImagePool(50)
        self.ssim = SSIM()

        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project_3d = Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.ego_diff = EWMA(momentum=0.98)

        # networks
        self.G = DispNet(self.opt)
        in_chs_D = 3 if self.opt.use_position_map else 1
        self.D = NLayerDiscriminator(in_chs_D, n_layers=3)

        # 就是算法图里的坐标
        if self.opt.use_position_map:
            h, w = self.opt.height, self.opt.width
            height_map = torch.arange(h).view(1, 1, h, 1).repeat(1, 1, 1, w) / (h - 1)
            width_map = torch.arange(w).view(1, 1, 1, w).repeat(1, 1, h, 1) / (w - 1)

            self.register_buffer('height_map', height_map, persistent=False)   # 将生成的矩阵存储到'height_map'，不用每次都生成
            self.register_buffer('width_map', width_map, persistent=False)

        # build day disp net
        self.day_dispnet = build_disp_net(
            Config.fromfile(osp.join('configs/', f'{self.opt.day_config}.yaml')),
            self.opt.day_check_point
        )

        

        # link to dataset
        self.data_link = opt.data_link

        # manual optimization
        self.automatic_optimization = False


    def forward(self, inputs):
        return self.G(inputs)
    
    def generate_gan_outputs(self, day_inputs, outputs):
        # (n, 1, h, w)
        night_disp = outputs['disp', 0, 0]
        with torch.no_grad():
            day_disp = self.day_dispnet(day_inputs)['disp', 0, 0]
        night_disp = night_disp / night_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        day_disp = day_disp / day_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        # image coordinates
        if self.opt.use_position_map:
            n = night_disp.shape[0]
            height_map = self.height_map.repeat(n, 1, 1, 1)
            width_map = self.width_map.repeat(n, 1, 1, 1)
        else:
            height_map = None
            width_map = None
        # return
        return day_disp, night_disp, height_map, width_map

    def compute_G_loss(self, night_disp, height_map, width_map):
        G_loss = 0.0
        #
        # Compute G loss
        freeze_model(self.D)
        if self.opt.use_position_map:
            fake_day = torch.cat([height_map, width_map, night_disp], dim=1)
        else:
            fake_day = night_disp
        G_loss += self.gan_loss(self.D(fake_day), True)

        return G_loss

    def compute_D_loss(self, day_disp, night_disp, height_map, width_map):
        D_loss = 0.0
        #
        # Compute D loss
        #
        unfreeze_model(self.D)
        if self.opt.use_position_map:
            real_day = torch.cat([height_map, width_map, day_disp], dim=1)
            fake_day = torch.cat([height_map, width_map, night_disp.detach()], dim=1)
        else:
            real_day = day_disp
            fake_day = night_disp.detach()
        # query
        fake_day = self.image_pool.query(fake_day)
        # compute loss
        D_loss += self.gan_loss(self.D(real_day), True)
        D_loss += self.gan_loss(self.D(fake_day), False)

        return D_loss



    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim_G, optim_D = self.optimizers()

        # tensorboard logger
        #logger = SummaryWriter(log_dir="data/log")
        logger = self.logger.experiment

        # get input data
        day_inputs = batch_data['day']
        night_inputs = batch_data['night']

        # outputs of G inputs
        outputs = self.G(night_inputs)

        # ------------outputs of segmentor night_inputs['color_aug', 0, 0] then to mask
        masks = []
        transformed_boxes = []
        for j in range(night_inputs['color_aug', 0, 0].shape[0]):
            night_inputs_in = torch.Tensor(night_inputs['color_aug', 0, 0][j].cpu()).float()  
            night_inputs_in = np.transpose(night_inputs_in,(1,2,0))         
            night_inputs_in = np.array(night_inputs_in)
            night_inputs_in = (night_inputs_in * 255).astype('uint8')   # 这里要么反归一化要么去找原图
            # box
            box_image = load_image(night_inputs_in)
            boxes_filt, pred_phrases = get_grounding_output(
                 self.sam_model, box_image, self.text_prompt, self.box_threshold, self.text_threshold, device=self.device2)
            # image
            self.predictor.set_image(night_inputs_in)
            # box
            H = night_inputs['color_aug', 0, 0][0].shape[1]
            W = night_inputs['color_aug', 0, 0][0].shape[2]
            if (boxes_filt.size(0)) >0:
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()
                transformed_box = self.predictor.transform.apply_boxes_torch(boxes_filt, night_inputs_in.shape[:2]).to(self.device2)  # (320,576)
                # mask         [instance,batch,h,w]
                mask, _, _ = self.predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_box.to(self.device2),
                    multimask_output = False,
                )
                mask = (torch.sum(mask,dim=0,keepdim=True) > 0).int()   # [1,shape,h,w]
                transformed_boxes.append(transformed_box.squeeze(1)) # list
                masks.append(mask.squeeze(1))   # [batch,h,w]  
            else:
                mask = np.zeros((1,1,H,W))
                mask_t = torch.Tensor(mask).to(self.device2)
                masks.append(mask_t.squeeze(1)) 
        masks = torch.stack(masks,dim=1)
        masks = masks.transpose(0,1)

        # loss for ego-motion
        disp_loss_dict = self.compute_disp_losses(night_inputs, outputs, masks)

        # generate outputs for gan
        day_disp, night_disp, height_map, width_map = self.generate_gan_outputs(day_inputs, outputs)

        # optimize G
        # compute loss
        G_loss = self.compute_G_loss(night_disp, height_map, width_map)  # loss_G
        disp_loss = sum(disp_loss_dict.values())   # loss_depth

        # log
        logger.add_scalar('train/disp_loss', disp_loss, self.global_step)
        logger.add_scalar('train/G_loss', G_loss, self.global_step)

        # optimize G
        G_loss = G_loss * self.opt.G_weight + disp_loss

        optim_G.zero_grad()
        self.manual_backward(G_loss)
        optim_G.step()

        #
        # optimize D
        #
        # compute loss
        D_loss = self.compute_D_loss(day_disp, night_disp, height_map, width_map)

        # log
        logger.add_scalar('train/D_loss', D_loss, self.global_step)

        D_loss = D_loss * self.opt.D_weight

        # optimize D
        optim_D.zero_grad()
        self.manual_backward(D_loss)   # loss.backward()
        optim_D.step()

        # # return
        # return G_loss + D_loss

    def training_epoch_end(self, outputs):
        """
        Step lr scheduler
        :param outputs:
        :return:
        """
        sch_G, sch_D = self.lr_schedulers()

        sch_G.step()
        sch_D.step()

        self.data_link.when_epoch_over()

    def configure_optimizers(self):
        optim_G = Adam(self.G.parameters(), lr=self.opt.learning_rate)
        optim_D = Adam(self.D.parameters(), lr=self.opt.learning_rate)

        sch_G = MultiStepLR(optim_G, milestones=[15], gamma=0.5)
        sch_D = MultiStepLR(optim_D, milestones=[15], gamma=0.5)

        return [optim_G, optim_D], [sch_G, sch_D]

    def get_color_input(self, inputs, frame_id, scale):      
        return inputs[("color_equ", frame_id, scale)] if self.opt.use_equ else inputs[("color", frame_id, scale)]

    def generate_images_pred(self, inputs, outputs, scale):   
        disp = outputs[("disp", 0, scale)]  
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)  
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)  
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):   
            T = outputs[("cam_T_cam", 0, frame_id)]   
            cam_points = self.backproject(depth, inputs["inv_K", 0])
            pix_coords = self.project_3d(cam_points, inputs["K", 0], T)  # [b,h,w,2]
            src_img = self.get_color_input(inputs, frame_id, 0)
            outputs[("color", frame_id, scale)] = F.grid_sample(src_img, pix_coords, padding_mode="border",
                                                                align_corners=False)    
        return outputs

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def get_static_mask(self, pred, target):
        # compute threshold  
        mask_threshold = self.ego_diff.running_val
        # compute diff
        diff = (pred - target).abs().mean(dim=1, keepdim=True)
        # compute mask
        static_mask = (diff > mask_threshold).float()
        # return
        return static_mask

    def compute_disp_losses(self, inputs, outputs, light_mask):
        loss_dict = {}
        light_mask = torch.where(light_mask == 1, 0, 1)  #.cpu().numpy()  

        for scale in self.opt.scales:
            """
            initialization 
            """
            disp = outputs[("disp", 0, scale)]
            target = self.get_color_input(inputs, 0, 0)  
            reprojection_losses = []

            """
            reconstruction  
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask  
            """
            use_static_mask = self.opt.use_static_mask
            # update ego diff  
            if use_static_mask:
                with torch.no_grad():
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = self.get_color_input(inputs, frame_id, 0)   

                        # get diff of two frames  difference  
                        diff = (pred - target).abs().mean(dim=1)     # Is-It
                        diff = torch.flatten(diff, 1)

                        # compute quantile 
                        quantile = torch.quantile(diff, self.opt.static_mask_quantile, dim=1)
                        mean_quantile = quantile.mean()

                        # update  
                        self.ego_diff.update(mean_quantile)

            # compute mask  
            for frame_id in self.opt.frame_ids[1:]:
                pred = self.get_color_input(inputs, frame_id, 0) 
                color_diff = self.compute_reprojection_loss(pred, target) 
                identity_reprojection_loss = color_diff + torch.randn(color_diff.shape).type_as(color_diff) * 1e-5   # 加噪

                # static mask
                if use_static_mask:
                    static_mask = self.get_static_mask(pred, target) 
                    identity_reprojection_loss *= static_mask 
                    identity_reprojection_loss *= light_mask  

                reprojection_losses.append(identity_reprojection_loss)   

            """
            minimum reconstruction loss    
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))   # It^-It
            reprojection_loss = torch.cat(reprojection_losses, 1)   # 列表，存储第12345张与0张的重建损失

            min_reconstruct_loss, _ = torch.min(reprojection_loss, dim=1)   # 表示每个像素点在所有 time 上的最小重建误差值
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

            """
            disp mean normalization   
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = get_smooth_loss(disp, self.get_color_input(inputs, 0, scale))
            loss_dict[('smooth_loss', scale)] = self.opt.disparity_smoothness * smooth_loss / (2 ** scale) / len(
                self.opt.scales)

        return loss_dict
    

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)


    def save_mask_data(output_dir, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump(json_data, f)
