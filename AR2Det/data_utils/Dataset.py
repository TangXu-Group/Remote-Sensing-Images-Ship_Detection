from __future__ import  absolute_import
from __future__ import  division
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import numpy as np
from .HRSCdataset import HRSCBboxDataset
import random
import torchvision.transforms as transforms
from math import pi
import cv2


def random_flip(img, bodies, centers, boxes, scores, y_random=False, x_random=False,
                return_param=False, copy=False):
    y_flip, x_flip = False, False
    
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        bodies = bodies[::-1,:]
        centers = centers[:,::-1,:]
        centers[1,:,:] = 512 - centers[1,:,:]
        
        boxes = boxes[:,::-1,:]
        boxes[1,:,:] = 512 - boxes[1,:,:]
        boxes[4,:,:] = -boxes[4,:,:]
        scores = scores[::-1,:]
        
        img = img[:, ::-1, :]
        
    if x_flip:
        bodies = bodies[:,::-1]
        centers = centers[:,:,::-1]
        centers[0,:,:] = 512 - centers[0,:,:]
        
        boxes = boxes[:,:,::-1]
        boxes[0,:,:] = 512 - boxes[0,:,:]
        boxes[4,:,:] = -boxes[4,:,:]
        scores = scores[:,::-1]
        
        img = img[:, :, ::-1]
        
    if copy:
        img = img.copy()
    if return_param:
        return img, bodies, centers, boxes, scores, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img, bodies, centers, boxes, scores

def resize_bbox(bboxes, count, in_size, out_size):
    if count > 0:
        bboxes = bboxes.copy()
        y_scale = out_size[0] / in_size[0]
        x_scale = out_size[1] / in_size[1]
        bboxes[:count, 0] = x_scale * bboxes[:count, 0]
        bboxes[:count, 1] = y_scale * bboxes[:count, 1]
        stx = bboxes[:count, 4].copy()
        bboxes[:count, 4] = np.arctan(np.tan(bboxes[:count, 4])*y_scale/x_scale)
        
        sty = (pi/2 - bboxes[:count, 4])*(bboxes[:count, 4]>0).astype(float) + ( (-pi/2) - bboxes[:count, 4])*(bboxes[:count, 4]<0).astype(float)
        sty1 = np.arctan(np.tan(sty)*y_scale/x_scale)
        
        bboxes[:count, 2] = x_scale * bboxes[:count, 2] * np.cos(stx)/np.cos(bboxes[:count, 4])
        bboxes[:count, 3] = x_scale * bboxes[:count, 3] * np.cos(sty)/np.cos(sty1) 
    return bboxes
        
def flip_bbox(bboxes, bboxes_count, size, y_flip=False, x_flip=False):
    H, W = size
    if bboxes_count > 0:
        if y_flip:
            bboxes[:, 1] = H - bboxes[:, 1]
            bboxes[:, 4] = - bboxes[:, 4]
        if x_flip:
            bboxes[:, 0] = W - bboxes[:, 0]
            bboxes[:, 4] = 3.14 - bboxes[:, 4]
    return bboxes
        
def pytorch_normalze(img):
    normalize = tvtsf.Normalize(mean=[0.281, 0.322, 0.310],
                                std=[0.196, 0.173, 0.165])
    img = normalize(t.from_numpy(img).float())
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):

    H, W, C = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = cv2.resize(img, (512 , 512)).transpose((2, 0, 1))
    normalize = pytorch_normalze
    return normalize(img), scale


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bboxes, bboxes_count, bodies, centers, boxes, scores= in_data
        H, W,_ = img.shape
        img, scale = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bboxes = resize_bbox(bboxes, bboxes_count, (H, W), (o_H, o_W))

        # horizontally flip
        img, bodies, centers, boxes, scores, params = random_flip(
            img, bodies, centers, boxes, scores, x_random=True, y_random=True,return_param=True)
        bboxes = flip_bbox(
            bboxes, bboxes_count, (o_H, o_W), x_flip=params['x_flip'],y_flip=params['y_flip'])
        
        return img, bboxes, scale, bodies, centers, boxes, scores


class Dataset:
    def __init__(self,length,data_dir):
        self.db = HRSCBboxDataset(data_dir,length,split = 'train')
        self.tsf = Transform()

    def __getitem__(self, idx):
        
        ori_img, boxes_with_class, bboxes_count, id, bodies, centers, boxes, scores = self.db.get_example(idx)
        img, bbox, _, bodies, centers, boxes, scores= self.tsf((ori_img, boxes_with_class, bboxes_count, bodies, centers, boxes, scores))

        return img.copy(), bbox.copy(), bboxes_count, id, bodies.copy(), centers.copy(), boxes.copy(), scores.copy()

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self,length,data_dir):
        self.db = HRSCBboxDataset(data_dir,length,split = 'test')

    def __getitem__(self, idx):
        ori_img, boxes_with_class, bboxes_count, id, bodies, centers, boxes, scores = self.db.get_example(idx)
        
        H, W,_ = ori_img.shape
        img,_ = preprocess(ori_img)
        _, o_H, o_W = img.shape
        bbox = resize_bbox(boxes_with_class, bboxes_count,(H, W), (o_H, o_W))
        return img.copy(), bbox.copy(),bboxes_count, id, bodies.copy(), centers.copy(), boxes.copy(), scores.copy()

    def __len__(self):
        return len(self.db)
    
class TestDataset_real:
    def __init__(self,length,data_dir,split = 'test'):
        self.db = HRSCBboxDataset(data_dir,length,split = split)

    def __getitem__(self, idx):
        ori_img, boxes_with_class, bboxes_count, id, bodies, centers, boxes, scores = self.db.get_example(idx)
        
        H, W,_ = ori_img.shape
        img,_ = preprocess(ori_img)
        return ori_img.copy(), img.copy(), boxes_with_class.copy(),bboxes_count, id, bodies.copy(), centers.copy(), boxes.copy(), scores.copy(), [H,W]

    def __len__(self):
        return len(self.db)
    
    
class Test_FPS_Dataset:
    def __init__(self,length,data_dir):
        self.db = HRSCBboxDataset(data_dir,length,split = 'fps_test')

    def __getitem__(self, idx):
        ori_img = self.db.get_example(idx)
        img,_ = preprocess(ori_img)
        return img.copy()

    def __len__(self):
        return len(self.db)
    
    
class ValDataset:
    def __init__(self,length,data_dir):
        self.db = HRSCBboxDataset(data_dir,length,split = 'val')

    def __getitem__(self, idx):
        ori_img, boxes_with_class, bboxes_count, id, bodies, centers, boxes, scores = self.db.get_example(idx)
        
        
        H, W,_ = ori_img.shape
        img,_ = preprocess(ori_img)
        _, o_H, o_W = img.shape
        bboxes = resize_bbox(boxes_with_class, bboxes_count,(H, W), (o_H, o_W))
        return img.copy(), bboxes.copy(),bboxes_count, id, bodies.copy(), centers.copy(), boxes.copy(), scores.copy()

    def __len__(self):
        return len(self.db)

class All:
    def __init__(self,length,data_dir):
        self.db = HRSCBboxDataset(data_dir,length,split = 'all')

    def __getitem__(self, idx):
        ori_img, boxes_with_class, bboxes_count, id = self.db.get_example(idx)
        
        H, W,_ = ori_img.shape
        img,_ = preprocess(ori_img)
        _, o_H, o_W = img.shape
        bboxes = resize_bbox(boxes_with_class, bboxes_count,(H, W), (o_H, o_W))
        return img.copy(), bboxes.copy(),bboxes_count, id

    def __len__(self):
        return len(self.db)
