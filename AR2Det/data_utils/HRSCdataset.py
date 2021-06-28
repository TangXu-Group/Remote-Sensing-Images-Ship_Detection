import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import pi
import torchvision.transforms as transforms

def read_image(path, dtype=np.float32, color=True,tran = 'train'):
    f = Image.open(path)
#     if tran == 'train':
#         f = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)(f)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    
    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        return img


class HRSCBboxDataset:
    
    '''for example:
    A = HRSCBboxDataset(data_dir = 'HRSC2016/HRSC2016.part01/HRSC2016/',split='test')
    dice = 452
    bb = A.get_example(dice)[0]
    plt.imshow(bb)'''
    
    def __init__(self,data_dir,length,split='train'):

        id_list_file = os.path.join(
            data_dir, 'ImageSets/'+split+'.txt')

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.split = split
        self.length = length
    def __len__(self):
        return len(self.ids)
    def get_example(self, i):
        id_ = self.ids[i]
        img_file = os.path.join(self.data_dir, 'FullDataSet/AllImages', id_ + '.bmp')
        img = read_image(img_file,tran = self.split)
        if self.split == 'fps_test': # For labels construction
            return img
        
        anno = ET.parse(os.path.join(self.data_dir, 'FullDataSet/Annotations', id_ + '.xml'))
        boxes_with_class = np.zeros((50,5+1+1+1)) #five parameters of bounding boxes + class_name + weather_diffcult + class_id
        count = 0 # the number of bounding boxes in this image.
        HRSC_Object = anno.find ('HRSC_Objects')
        for obj in HRSC_Object.findall ('HRSC_Object'):
#             if int(obj.find('difficult').text) == 1:
#                 continue
            boxes_with_class[count,:5] = np.array([float(obj.find(tag).text) for tag in ('mbox_cx', 'mbox_cy', 'mbox_w', 'mbox_h','mbox_ang')])
            name = obj.find('Class_ID').text
            boxes_with_class[count,5] = int(name)-100000001
            boxes_with_class[count,6] = int(obj.find('difficult').text)
            boxes_with_class[count,7] = int(obj.find('Class_ID').text)
            count +=1
        
        if self.split == 'all': # For labels construction
            return img, boxes_with_class,count, id_
        
        bodies = os.path.join(self.data_dir, 'Labels'+str(self.length)+'/bodies', id_ + '.npy')
        bodies = np.load(bodies)
        centers = os.path.join(self.data_dir, 'Labels'+str(self.length)+'/centers', id_ + '.npy')
        centers = np.load(centers)
        boxes = os.path.join(self.data_dir, 'Labels'+str(self.length)+'/boxes', id_ + '.npy')
        boxes = np.load(boxes)
        scores = os.path.join(self.data_dir, 'Labels'+str(self.length)+'/scores', id_ + '.npy')
        scores = np.load(scores)
        return img, boxes_with_class,count, id_, bodies, centers, boxes, scores

    __getitem__ = get_example
    
VOC_BBOX_LABEL_NAMES = [
    'ship',
    'aircraft carrier',
    'warcraft',
    'merchant ship',
    'Nimitz',
    'Enterprise',
    'Arleigh Burke',
    'WhidbeyIsland',
    'Perry',
    'Sanantonio',
    'Ticonderoga',
    'Kitty Hawk',
    'Kuznetsov',
    'Abukuma',
    'Austen',
    'Tarawa',
    'Blue Ridge',
    'Container',
    'OXo|--)',
    'Car carrier([]==[])',
    '',
    'Hovercraft',
    '',
    'yacht',
    'CntShip(_|.--.--|_]=',
    'Cruise',
    'submarine',
    'lute',
    'Medical',
    'Car carrier(======|',
    'Ford-class',
    'Midway-class',
    'Invincible-class']