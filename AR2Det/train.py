gpu_id = '0'
resnet_type = '34'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

import torch
import time
from tqdm import tqdm

from torch.utils.data import DataLoader
from data_utils.Dataset import Dataset,TestDataset,ValDataset,Test_FPS_Dataset,TestDataset_real

from tra_val.evaluation import *
from tra_val.loss import *
from tra_val.tra_val import *
from model.FARN import FARN
data_dir= 'HRSC2016/'


train_batch_size = 1 # default:2
test_batch_size = 1 # fixed:1
val_batch_size = 1 # fixed:1
fps_bath_size = 8 # default:1

train_dataset = Dataset(data_dir) #train dataset with bbox
tra_Dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True)

test_dataset = TestDataset(data_dir) #test dataset with bbox
tes_Dataloader = DataLoader(test_dataset, test_batch_size, shuffle=False,
                                               num_workers=8, pin_memory=True)

val_dataset = ValDataset(data_dir) #val dataset with bbox
val_Dataloader = DataLoader(val_dataset, val_batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True)

test_fps_dataset = Test_FPS_Dataset(data_dir) #FPS test only with image
test_fps_Dataloader = DataLoader(test_fps_dataset, fps_bath_size, shuffle=True,
                                               num_workers=8, pin_memory=True)

testDataset_real = TestDataset_real(data_dir) #test dateset without resize_bbox
tes_realDataloader = DataLoader(testDataset_real, test_batch_size, shuffle=False,
                                               num_workers=8, pin_memory=True)

#pixels coordinates
pixel_coordinates = torch.zeros(128,128,2).cuda()
print('pixel_coordinates constructing')
for i in tqdm(range(128)):
    for ii in range(128):
        pixel_coordinates[i,ii,:] = torch.tensor([ii,i])*4

#model construction
# model = FARN(resnet_type='resnet101', trans_channel_num=256, resnet_layer=[2048, 1024,512, 256], ).cuda()
model = FARN(resnet_type='resnet'+resnet_type).cuda()
# model = FARN().cuda()                                                                                                              
optimizer =torch.optim.Adam(model.parameters(),lr=0.0001)

#load trained model
# model_trained = FARN(resnet_type='resnet34').cuda()
# model_trained.load_state_dict(torch.load('./g0_FARN_07map_0.8751_date_20200814_20_39_13'))

#training logging
Log_path = 'output/g'+gpu_id+'_FARN_'+resnet_type+str(time.strftime("_date_%Y%m%d"))+str(time.strftime("_%H_%M_%S.txt"))


bestmap_07 = 0
bestmap_12 = 0
t = 0.5
for epoch in range(5000):
    if epoch < 10:
        validate_interval = 3
    else:
        validate_interval = 1
    Train_FARN(model,
               epoch,
               tra_Dataloader,
               optimizer,
               train_batch_size,
               pixel_coordinates,
               bodies_theshold=t,
               coeff_dxdy=0.1,
               coeff_theta=10,
               visloss_per_iter=tra_Dataloader.__len__()/1,
               if_iou = False)
    if (epoch + 1) % validate_interval == 0:
        MAP_07, MAP_12 = Validate_FARN(model,
                                       epoch,
                                       tes_Dataloader,
                                       pixel_coordinates,
                                       Log_path,
                                       bestmap_07,
                                       bestmap_12,
                                       fliter_theshold=t,
                                       scores_theshold=0,
                                       nms_theshold=0.7,
                                       nms_saved_images_num=20,
                                       plot_PR=False)
        if MAP_07 >= bestmap_07:
            bestmap_07 = MAP_07
            if MAP_07 > 0.80:
                torch.save(model.state_dict(), 'checkpoints/g'+gpu_id+'_FARN_'+resnet_type+'_07map_'+str(round(bestmap_07, 4))+str(time.strftime("_date_%Y%m%d"))+str(time.strftime("_%H_%M_%S")))
        if MAP_12 >= bestmap_12:
            bestmap_12 = MAP_12
#             if MAP_12 > 0.80:
#                 torch.save(model.state_dict(), 'checkpoints/FARN_'+'12map_'+str(round(bestmap_12, 4))+str(time.strftime("_date_%Y%m%d"))+str(time.strftime("_%H_%M_%S")))