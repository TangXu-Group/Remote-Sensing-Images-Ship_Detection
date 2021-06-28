import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import time
from data_utils.Dataset import Dataset,TestDataset,ValDataset,All

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
data_dir= 'HRSC2016/'

def make_bodies_label(use_dataset, size, static_pixel_1ocation, center_size = [5,5], shrink_ratio = 0.7):
    print('bodies_label constructing:')
    time.sleep(0.5)
    dynamic_pixel_1ocation = torch.zeros(size,size,2)
    for index in tqdm(range(use_dataset.__len__())):
        data = use_dataset[index]
        bboxes = torch.tensor(data[1])
        data_id = data[3]

        bboxes_count = data[2] #bbox 数量
        m=torch.zeros(size,size) #mask

        for bbox_id in range(bboxes_count):
            bbox_x,bbox_y,bbox_w,bbox_h = bboxes[bbox_id,:4]/(512/size)
            bbox_theta = bboxes[bbox_id,4]

            dynamic_pixel_1ocation[:,:,:] = static_pixel_1ocation[:,:,:]
            dynamic_pixel_1ocation[:,:,0:2] = dynamic_pixel_1ocation[:,:,0:2]-torch.Tensor([bbox_x,bbox_y])
            X = dynamic_pixel_1ocation[:,:,0]*np.cos(bbox_theta).float() + dynamic_pixel_1ocation[:,:,1]*np.sin(bbox_theta).float()
            Y = dynamic_pixel_1ocation[:,:,1]*np.cos(bbox_theta).float() - dynamic_pixel_1ocation[:,:,0]*np.sin(bbox_theta).float()
            dynamic_pixel_1ocation[:,:,0] = X[:,:]
            dynamic_pixel_1ocation[:,:,1] = Y[:,:]

            rm = (m==0).float()
            
            #for masking ship boxes prediction
            m1 = dynamic_pixel_1ocation[:,:,0]<np.min((int(center_size[0]/2)+1,(bbox_w/2).float()))
            m2 = dynamic_pixel_1ocation[:,:,0]>-np.min((int(center_size[0]/2)+1,(bbox_w/2).float()))
            m3 = dynamic_pixel_1ocation[:,:,1]<np.min((int(center_size[1]/2)+1,(bbox_h/2).float()))
            m4 = dynamic_pixel_1ocation[:,:,1]>-np.min((int(center_size[1]/2)+1,(bbox_h/2).float()))
            m += (m1*m2*m3*m4).float()

            #for masking ship bodies prediction
            m1 = dynamic_pixel_1ocation[:,:,0]<(bbox_w/2).float()*shrink_ratio
            m2 = dynamic_pixel_1ocation[:,:,0]>-(bbox_w/2).float()*shrink_ratio
            m3 = dynamic_pixel_1ocation[:,:,1]<(bbox_h/2).float()*shrink_ratio
            m4 = dynamic_pixel_1ocation[:,:,1]>-(bbox_h/2).float()*shrink_ratio
            m += (m1*m2*m3*m4).float()*rm

            #for masking ship centers prediction
            m1 = dynamic_pixel_1ocation[:,:,0]<(bbox_w/2).float()
            m2 = dynamic_pixel_1ocation[:,:,0]>-(bbox_w/2).float()
            m3 = dynamic_pixel_1ocation[:,:,1]<(bbox_h/2).float()
            m4 = dynamic_pixel_1ocation[:,:,1]>-(bbox_h/2).float()
            m += (m1*m2*m3*m4).float()*rm

        if torch.max(m)!=0:
            m = m/torch.max(m)
#         plt.imshow(m)
#         plt.show()
        np.save('HRSC2016/Labels/bodies/'+data_id+'.npy',(m).float())
    
    
def make_centers_label(use_dataset, size, static_pixel_1ocation, shrink_ratio = 1):
    print('centers_label constructing:')
    time.sleep(0.5)
    dynamic_pixel_1ocation = torch.zeros(size,size,2)
    dynamic_bboxes_label = torch.zeros(2,size,size)
    for index in tqdm(range(use_dataset.__len__())):
        data = use_dataset[index]
        bboxes = torch.tensor(data[1])
        data_id = data[3]
        
        bboxes_count = data[2] #bbox 数量
        m=torch.zeros(size,size) #mask

        if (bboxes_count)>0 :
            bboxes[:,:4] = bboxes[:,:4]/(512/size)
            static_bboxes_label = torch.zeros(2,size ,size )
            for bbox_id in range(bboxes_count):
                bbox_x,bbox_y,bbox_w,bbox_h = bboxes[bbox_id,:4]
                bbox_theta = bboxes[bbox_id,4]

                dynamic_pixel_1ocation[:,:,:] = static_pixel_1ocation[:,:,:]
                dynamic_pixel_1ocation[:,:,0:2] = dynamic_pixel_1ocation[:,:,0:2]-torch.Tensor([bbox_x,bbox_y])
                X = dynamic_pixel_1ocation[:,:,0]*np.cos(bbox_theta).float() + dynamic_pixel_1ocation[:,:,1]*np.sin(bbox_theta).float()
                Y = dynamic_pixel_1ocation[:,:,1]*np.cos(bbox_theta).float() - dynamic_pixel_1ocation[:,:,0]*np.sin(bbox_theta).float()
                dynamic_pixel_1ocation[:,:,0] = X[:,:]
                dynamic_pixel_1ocation[:,:,1] = Y[:,:]

                m1 = dynamic_pixel_1ocation[:,:,0]<(bbox_w/2).float()*shrink_ratio
                m2 = dynamic_pixel_1ocation[:,:,0]>-(bbox_w/2).float()*shrink_ratio
                m3 = dynamic_pixel_1ocation[:,:,1]<(bbox_h/2).float()*shrink_ratio
                m4 = dynamic_pixel_1ocation[:,:,1]>-(bbox_h/2).float()*shrink_ratio
                m += (m1*m2*m3*m4).float()

                for parameter in range(2):
                    dynamic_bboxes_label[parameter,:,:] = bboxes[bbox_id,parameter]*(512/size)

                static_bboxes_label += (m1*m2*m3*m4).float() * dynamic_bboxes_label[:,:,:] * (m==1).float()

            static_bboxes_label = static_bboxes_label+(static_bboxes_label==0).float() #0------->1
        else :
            static_bboxes_label = torch.ones(2,size,size) #0------->1
#         plt.imshow(static_bboxes_label[0])
#         plt.show()
        np.save('HRSC2016/Labels/centers/'+data_id+'.npy',static_bboxes_label)
    
    
def make_boxes_label(use_dataset, size, static_pixel_1ocation, center_size = [5,5]):
    print('boxes_label constructing:')
    time.sleep(0.5)
    dynamic_pixel_1ocation = torch.zeros(size,size,2)
    dynamic_bboxes_label = torch.zeros(5,size,size)
    for index in tqdm(range(use_dataset.__len__())):
        data = use_dataset[index]
        bboxes = torch.tensor(data[1])
        data_id = data[3]

        bboxes_count = data[2] #bboxes 数量
        m=torch.zeros(size,size) #mask

        if (bboxes_count)>0 :
            bboxes[:,:4] = bboxes[:,:4]/(512/size)
            static_bboxes_label = torch.zeros(5,size ,size )
            for bbox_id in range(bboxes_count):
                bbox_x,bbox_y,bbox_w,bbox_h = bboxes[bbox_id,:4]
                bbox_theta = bboxes[bbox_id,4]

                dynamic_pixel_1ocation[:,:,:] = static_pixel_1ocation[:,:,:]
                dynamic_pixel_1ocation[:,:,0:2] = dynamic_pixel_1ocation[:,:,0:2]-torch.Tensor([bbox_x,bbox_y])
                X = dynamic_pixel_1ocation[:,:,0]*np.cos(bbox_theta).float() + dynamic_pixel_1ocation[:,:,1]*np.sin(bbox_theta).float()
                Y = dynamic_pixel_1ocation[:,:,1]*np.cos(bbox_theta).float() - dynamic_pixel_1ocation[:,:,0]*np.sin(bbox_theta).float()
                dynamic_pixel_1ocation[:,:,0] = X[:,:]
                dynamic_pixel_1ocation[:,:,1] = Y[:,:]
                m1 = dynamic_pixel_1ocation[:,:,0]<np.min((int(center_size[0]/2)+1,(bbox_w/2).float()))
                m2 = dynamic_pixel_1ocation[:,:,0]>-np.min((int(center_size[0]/2)+1,(bbox_w/2).float()))
                m3 = dynamic_pixel_1ocation[:,:,1]<np.min((int(center_size[1]/2)+1,(bbox_h/2).float()))
                m4 = dynamic_pixel_1ocation[:,:,1]>-np.min((int(center_size[1]/2)+1,(bbox_h/2).float()))
                m += (m1*m2*m3*m4).float()

                for parameter in range(4):
                    dynamic_bboxes_label[parameter,:,:] = bboxes[bbox_id,parameter]*(512/size)
                dynamic_bboxes_label[4,:,:] = bboxes[bbox_id,4]

                static_bboxes_label += (m1*m2*m3*m4).float() * dynamic_bboxes_label[:,:,:] * (m==1).float()

            static_bboxes_label = static_bboxes_label+(static_bboxes_label==0).float() #0------->1
        else :
            static_bboxes_label = torch.ones(5,size,size) #0------->1

        np.save('HRSC2016/Labels/boxes/'+data_id+'.npy',static_bboxes_label)
        
        
def make_scores_label(use_dataset, size, static_pixel_1ocation,center_size = [5,5],expansion_steps = 30):
    print('scores_label constructing:')
    time.sleep(0.5)
    dynamic_pixel_1ocation = torch.zeros(size,size,2)
    for index in tqdm(range(use_dataset.__len__())):
        data = use_dataset[index]
        bboxes = torch.tensor(data[1])
        data_id = data[3]

        bboxes_count = data[2] #bbox 数量
        m=torch.zeros(size,size) #mask

        for bbox_id in range(bboxes_count):
            bbox_x,bbox_y,bbox_w,bbox_h = bboxes[bbox_id,:4]/(512/size)
            bbox_theta = bboxes[bbox_id,4]

            dynamic_pixel_1ocation[:,:,:] = static_pixel_1ocation[:,:,:]
            dynamic_pixel_1ocation[:,:,0:2] = dynamic_pixel_1ocation[:,:,0:2]-torch.Tensor([bbox_x,bbox_y])
            X = dynamic_pixel_1ocation[:,:,0]*np.cos(bbox_theta).float() + dynamic_pixel_1ocation[:,:,1]*np.sin(bbox_theta).float()
            Y = dynamic_pixel_1ocation[:,:,1]*np.cos(bbox_theta).float() - dynamic_pixel_1ocation[:,:,0]*np.sin(bbox_theta).float()
            dynamic_pixel_1ocation[:,:,0] = X[:,:]
            dynamic_pixel_1ocation[:,:,1] = Y[:,:]

            rm = (m==0).float()
            for long in range(expansion_steps+1):
                stridew = (bbox_w/2 - (int(center_size[0]/2)+1))/expansion_steps
                strideh = (bbox_h/2 - (int(center_size[1]/2)+1))/expansion_steps
                f1 = stridew * long
                f2 = strideh * long
                if bbox_w/2 <=int(center_size[0]/2)+1:
                    f1 = 1
                if bbox_h/2 <=int(center_size[1]/2)+1:
                    f2 = 1
                m1 = dynamic_pixel_1ocation[:,:,0]<= (bbox_w/2).float()-f1
                m2 = dynamic_pixel_1ocation[:,:,0]>= -(bbox_w/2).float()+f1
                m3 = dynamic_pixel_1ocation[:,:,1]<= (bbox_h/2).float()-f2
                m4 = dynamic_pixel_1ocation[:,:,1]>= -(bbox_h/2).float()+f2
                m += (m1*m2*m3*m4).float()*rm

        if torch.max(m)!=0:
            m = m/torch.max(m)
#         plt.imshow(m)
#         plt.show()
        np.save('HRSC2016/Labels/scores/'+data_id+'.npy',(m).float())
    
if __name__=="__main__":
    train_dataset = Dataset(data_dir)
    test_dataset = TestDataset(data_dir)
    val_dataset = ValDataset(data_dir)
    all_dataset = All(data_dir)

    use_dataset = all_dataset  #the used dataset
    label_size = 128  #label_size
    center_size = [5,5]
    shrink_ratio_bodies = 0.7
    shrink_ratio_centers = 1
    expansion_steps = 30
    
    #initialize pixel's location
    static_pixel_1ocation = torch.zeros(label_size,label_size,2)
    for i in range(label_size):
        for ii in range(label_size):
            static_pixel_1ocation[i,ii,:] = torch.tensor([ii,i])

    if 1-os.path.exists('./HRSC2016/Labels'):
        os.mkdir('./HRSC2016/Labels')
    if 1-os.path.exists('./HRSC2016/Labels/bodies'):
        os.mkdir('./HRSC2016/Labels/bodies')
    if 1-os.path.exists('./HRSC2016/Labels/centers'):
        os.mkdir('./HRSC2016/Labels/centers')
    if 1-os.path.exists('./HRSC2016/Labels/boxes'):
        os.mkdir('./HRSC2016/Labels/boxes')
    if 1-os.path.exists('./HRSC2016/Labels/scores'):
        os.mkdir('./HRSC2016/Labels/scores')
    
    make_bodies_label(use_dataset, label_size, static_pixel_1ocation, center_size, shrink_ratio_bodies)
    make_centers_label(use_dataset, label_size, static_pixel_1ocation, shrink_ratio_centers)
    make_boxes_label(use_dataset, label_size, static_pixel_1ocation, center_size)
    make_scores_label(use_dataset, label_size, static_pixel_1ocation,center_size, expansion_steps)