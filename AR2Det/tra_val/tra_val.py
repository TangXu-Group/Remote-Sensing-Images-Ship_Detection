import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import math
import cv2 as cv
import random
import time
import datetime
import gc #clean memory
from numpy import trapz # P-R
from .draw_bbox import draw_bbox

from .evaluation import *
from .loss import *
from .tra_val import *

def Train_FARN(model, epoch, tra_Dataloader, optimizer, batch_size, pixel_coordinates, bodies_theshold = 0.5, coeff_dxdy = 5, visloss_per_iter = 140, if_iou = False, S_set = False):
    lossvis_bodies = []
    lossvis_centers = []
    lossvis_boxes = []
    lossvis_scores = []
    lossvis_sum = []
    for iteration,(img,bboxes,bboxes_count,id,bodies, centers, boxes, scores) in enumerate(tqdm(tra_Dataloader,desc='train')):
        img,bboxes,bboxes_count,bodies, centers, boxes, scores = img.cuda(),bboxes.cuda(),bboxes_count.cuda(),bodies.cuda(), centers.cuda(), boxes.cuda(), scores.cuda()
        mask,features = model(img)
        background_ship_ratio = torch.sum((bodies>0).float(),dim=[1,2])/torch.sum((bodies==0).float(),dim=[1,2])
        batch_size = background_ship_ratio.shape[0]
        random_select = torch.zeros(batch_size,128,128).cuda()
        
#         shrunk R
#         for batch_num in range(batch_size):
#             random_select[batch_num] = (torch.randint(low=0,high =int(1/float(background_ship_ratio[batch_num])),size =[128,128])==1).float().cuda()
#             if int(1/float(background_ship_ratio[batch_num])) < 1:
#                 print('your background is soo small, so, for this image, you needn\'t to remove the background.')

        if if_iou :
            iou_ = torch.zeros(batch_size,128,128).cuda()
            Pr_bboxes = []
            Pr_bboxes.append(pixel_coordinates[:,:,0].unsqueeze(0)+features[:,0,:,:])
            Pr_bboxes.append(pixel_coordinates[:,:,1].unsqueeze(0)+features[:,1,:,:])
            Pr_bboxes.append(features[:,2,:,:])
            Pr_bboxes.append(features[:,3,:,:])
            Pr_bboxes.append(features[:,4,:,:])
            Pr_bboxes = torch.stack(Pr_bboxes).permute(1,2,3,0) #batch, w, h, (xywhtheta)
            for batch in range(batch_size):
#                 iou_[batch] = bbox_iou_scores1_n(Pr_bboxes[batch],bboxes[batch,:int(bboxes_count[batch]),:5], (bodies>0).float()[batch])
                iou_[batch] = bbox_iou_scores1_1(Pr_bboxes[batch],centers[batch].permute(1,2,0), (bodies>0).float()[batch])
            scores = iou_
        
#         if S_set == False:
#             random_select = (mask[:,0,:,:]>0).float()
            
        '''♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦scores♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦'''
        positive = (scores>0).float()
#         random_select = ((random_select + positive)>0).float()
        scores_positive1 = torch.log((scores+0.000001)/(mask[:,0,:,:]+0.000001))*(mask[:,0,:,:]<scores).float()*positive
        scores_positive2 = torch.log((mask[:,0,:,:]+0.000001)/(scores+0.000001))*(mask[:,0,:,:]>scores).float()*positive
        scores_positive = scores_positive1 + scores_positive2
        scores_negative = torch.log(1/(1-mask[:,0,:,:]+0.000001))*(1-positive)
        loss_scores = torch.sum(scores_positive + scores_negative)/(128*128*batch_size)

        '''♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦confidences♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦'''
        eliminate = (mask[:,1,:,:]>bodies_theshold).float()
        bodies_positive = torch.log(1/(mask[:,1,:,:]+0.000001))*(bodies>0.4).float()
        bodies_negative = torch.log(1/(1-(mask[:,1,:,:]*eliminate)+0.000001))*(1-(bodies>0).float())
        bodies_sum = torch.sum((bodies>0.4).float())+torch.sum(eliminate*(1-(bodies>0).float()))
        loss_bodies = torch.sum(bodies_positive + bodies_negative)/(bodies_sum)

        '''♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦boxes♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦'''
        tx_box = (pixel_coordinates[:,:,0]+features[:,0] - boxes[:,0])/coeff_dxdy *(1+torch.abs(torch.sin(boxes[:,4])))
        ty_box = (pixel_coordinates[:,:,1]+features[:,1] - boxes[:,1])/coeff_dxdy *(1+torch.abs(torch.cos(boxes[:,4])))
        tw_box = logfunction(features[:,2],boxes[:,2])
        th_box = logfunction(features[:,3],boxes[:,3])
        ttheta_box = (features[:,4]-boxes[:,4])*(boxes[:,2]/boxes[:,3])
        locl = smoothL1(tx_box,ty_box,tw_box,th_box,ttheta_box) *(bodies==1).float()
        loss_boxes = torch.sum(locl)/torch.sum((bodies==1).float())

        '''♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦centers♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦'''
        tx_center = (pixel_coordinates[:,:,0]+features[:,5] - centers[:,0])/coeff_dxdy * (1+torch.abs(torch.sin(centers[:,4])))
        ty_center = (pixel_coordinates[:,:,1]+features[:,6] - centers[:,1])/coeff_dxdy * (1+torch.abs(torch.cos(centers[:,4])))
        loss_centers = torch.sum(smoothL1_2(tx_center,ty_center)*(bodies>0).float())/torch.sum((bodies>0).float())

        '''♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦SUM♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦♦'''

        loss_bodies = loss_bodies
        loss_centers = loss_centers
        loss_boxes = loss_boxes
        loss_scores = loss_scores
        loss_sum = loss_bodies + loss_centers + loss_boxes + loss_scores

        lossvis_bodies.append(float(loss_bodies.cpu().detach().numpy()))
        lossvis_centers.append(float(loss_centers.cpu().detach().numpy()))
        lossvis_boxes.append(float(loss_boxes.cpu().detach().numpy()))
        lossvis_scores.append(float(loss_scores.cpu().detach().numpy()))
        lossvis_sum.append(float(loss_sum.cpu().detach().numpy()))

#       backward
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        if (iteration+1) % visloss_per_iter == 0:
            print('')
            print('epoch:',epoch,'         sum loss:',round(np.mean(lossvis_sum), 5))
            print('epoch:',epoch,' bodies loss:',round(np.mean(lossvis_bodies), 5))
            print('epoch:',epoch,'centers loss:',round(np.mean(lossvis_centers), 5))
            print('epoch:',epoch,'  boxes loss:',round(np.mean(lossvis_boxes), 5))
            print('epoch:',epoch,' scores loss:',round(np.mean(lossvis_scores), 5))
    return round(np.mean(lossvis_sum), 5)

def Validate_FARN(model, epoch, tes_Dataloader, pixel_coordinates, Log_path, bestmap_07, bestmap_12,IOU_theshold = 0.5, fliter_theshold = 0.5, scores_theshold = 0, nms_theshold = 0.7, nms_saved_images_num = 20, plot_PR = False, if_relative_c = True):
    '''♦♦♦♦♦♦♦♦♦♦Val♦♦♦♦♦♦♦♦♦♦'''
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for i,(ori_img, img,bboxes,bboxes_count,_,bodies, centers, boxes, scores, (H,W)) in enumerate(tqdm(tes_Dataloader,desc='val')):
        img,bboxes,bboxes_count,bodies, centers, boxes, scores = img.cuda(),bboxes.cuda(),bboxes_count.cuda(),bodies.cuda(), centers.cuda(), boxes.cuda(), scores.cuda()
        H,W = float(H),float(W)
        o_H,o_W = 512,512
        
        mask,features = model(img)
        mask,features = mask.detach(), features.detach()
        
        if if_relative_c:
            pred_xy_fea_ = (features[0,5:7].round().float()+pixel_coordinates[:,:,0:2].permute(2,0,1))/4
            pred_xy_fea_ = (pred_xy_fea_>127).float()*127 + pred_xy_fea_*(pred_xy_fea_<=127).float()
            pred_xy_fea_ = pred_xy_fea_*(pred_xy_fea_>0).float()  
        else:
            pred_xy_fea_ = ((features[0,5:7].round().float())/4)
            
        mm = (mask[0,1]>fliter_theshold).float()
        pred_xy_fea = pred_xy_fea_ * mm
        pred_centers = torch.zeros(128,128).cuda()
        pred_centers[pred_xy_fea[1].long(),pred_xy_fea[0].long()] = 1
        pred_centers[0,0]=0

        sum_centers = torch.sum(pred_centers).int()
        resl = torch.cat((pred_xy_fea_[0].unsqueeze(2),pred_xy_fea_[1].unsqueeze(2),mask[0,0].unsqueeze(2)),2)
        resl = (resl*(pred_centers).unsqueeze(2)).view(-1,3)
        resl = resl[torch.argsort(resl[:,0],descending=True)]
        
        pred_centers = np.zeros((128,128))
        for xyscore_pixel in resl[:sum_centers]:
            if pred_centers[xyscore_pixel[1].long(),xyscore_pixel[0].long()] < xyscore_pixel[2]:
                pred_centers[xyscore_pixel[1].long(),xyscore_pixel[0].long()] = xyscore_pixel[2]
        pred_centers = torch.tensor(pred_centers).cuda().float()

        Pr_bboxes = []
        if if_relative_c:
            Pr_bboxes.append(pixel_coordinates[:,:,0]+features[0,0,:,:])
            Pr_bboxes.append(pixel_coordinates[:,:,1]+features[0,1,:,:])
        else:
            Pr_bboxes.append(features[0,0,:,:])
            Pr_bboxes.append(features[0,1,:,:])
        Pr_bboxes.append(features[0,2,:,:])
        Pr_bboxes.append(features[0,3,:,:])
        Pr_bboxes.append(features[0,4,:,:])
        Pr_bboxes.append(pred_centers)
        Pr_bboxes = torch.stack(Pr_bboxes).permute(1,2,0).view(-1,6)
        Pr_bboxes = Pr_bboxes[torch.argsort(Pr_bboxes[:,5])] #sorting
        Pr_bboxes = Pr_bboxes[-torch.sum(Pr_bboxes[:,5]>scores_theshold):] #selecting pixels
        Pr_bboxes = resize_bbox(np.array(Pr_bboxes.cpu()), (o_H, o_W), (H, W))
        Pr_bboxes = torch.FloatTensor(Pr_bboxes)
        
        Gt_bboxes = np.array(bboxes[0,:bboxes_count,:5].cpu().unsqueeze(0))

        if torch.sum(Pr_bboxes[:,5]>scores_theshold)!=0:
            keep = rotate_gpu_nms(np.array(Pr_bboxes.cpu()),nms_theshold)
            predicted_all_bboxes = Pr_bboxes[:,:6].cpu()
            predicted_bboxes_id = keep[:nms_saved_images_num]
            predicted_bboxes = np.array(predicted_all_bboxes[predicted_bboxes_id].cpu().unsqueeze(0))

            pred_bboxes_ = list(predicted_bboxes[:,:,:5])
            pred_labels_ = list(np.zeros((1,len(predicted_bboxes[0]))))
            pred_scores_ = list(predicted_bboxes[:,:,5])
        else:
            preiou = np.zeros((0,0,0))
            pred_bboxes_ =list(np.zeros((1,0,5)))
            pred_labels_ = list(np.zeros((1,0)))
            pred_scores_ = list(np.zeros((1,0)))

        gt_bboxes += list(Gt_bboxes[:,:,:5])
        gt_labels += list(np.zeros((1,len(Gt_bboxes[0]))))
        gt_difficults += list(np.zeros((1,len(Gt_bboxes[0]))))
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    #MAP VOC 07
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True,iou_thresh = IOU_theshold)
    MAP_07 = result[0]['map']
    
    #MAP VOC 12
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=False,iou_thresh = IOU_theshold)
    MAP_12 = result[0]['map']

    print(('Epoch:'+str(epoch)+'   07_MAP:'+str(MAP_07)+'   12_MAP:'+str(MAP_12)+'   Best07_Map:'+str(bestmap_07)+'   Best12_Map:'+str(bestmap_12)+'\n'))
    f = open(Log_path,'a')
    f.write('Epoch:'+str(epoch)+'   07_MAP:'+str(MAP_07)+'   12_MAP:'+str(MAP_12)+'   Best07_Map:'+str(bestmap_07)+'   Best12_Map:'+str(bestmap_12)+'\n')
    f.close()

    if plot_PR:
        PR = np.zeros((len(list(result[1][0])),2))
        PR[:,0] = np.array(list(result[1][0]))
        PR[:,1] = np.array(list(result[2][0]))
        PR = PR[PR[:,1].argsort()] 
        for i in tqdm(range(len(list(result[1][0])))):
            for ii in range(i):
                if PR[ii,0]<PR[i,0]:
                    PR[ii,0]=PR[i,0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlim=[0, 1], ylim=[0, 1], title='PR',
               ylabel='Percision', xlabel='Recall')
        plot1 = plt.plot(PR[:,1], PR[:,0], 'r',label='original values')
        plt.show()
    return MAP_07, MAP_12

def FPS_test(model_trained, test_fps_Dataloader, pixel_coordinates, test_epoch = 5, fliter_theshold = 0.5, scores_theshold = 0, nms_theshold = 0.7, nms_saved_images_num = 20, get_score = False):
    image_count = 0
    print('counting the number of images')
    for i,(img) in enumerate(test_fps_Dataloader):
        batch_size = img.shape[0]
        image_count+=batch_size
    print('number of images:',image_count)
    
    start = datetime.datetime.now()
    print('testing FPS')
    for i in tqdm(range(test_epoch)):
        for i,(img) in enumerate(test_fps_Dataloader):
            batch_size = img.shape[0]
            mask,features = model_trained(img.cuda())
            mask,features = mask.detach(), features.detach()
            mm = (mask[:,1].detach()>0.5).float()
            
            pred_xy_fea_ = (((features[:,5:7].round().float())+pixel_coordinates[:,:,0:2].permute(2,0,1).unsqueeze(0))/4)
            pred_xy_fea_ = pred_xy_fea_*(pred_xy_fea_<128).float()*(pred_xy_fea_>0).float()
            pred_xy_fea_ = (pred_xy_fea_ * mm.unsqueeze(1))

            pred_centers = torch.zeros(batch_size,128,128).cuda()
            for batch in range(batch_size):
                pred_centers[batch,pred_xy_fea_[batch,1].long(),pred_xy_fea_[batch,0].long()] = 1
            pred_centers[:,0,0]=0

            if 1-get_score:
                mm = (pred_centers==1).float()
                pred_xy_fea = (pred_xy_fea_ * mm.unsqueeze(1)).long()

                pred_centers = torch.zeros(batch_size,128,128).cuda()
                for batch in range(batch_size):
                    pred_centers[batch,pred_xy_fea[batch,1],pred_xy_fea[batch,0]] = 1
                pred_centers[:,0,0]=0
            else:
                sum_centers = torch.sum(pred_centers[:].view(batch_size,-1),dim=1).int()
                resl = torch.cat((pred_xy_fea_[:,0].unsqueeze(3),pred_xy_fea_[:,1].unsqueeze(3),mask[:,0].unsqueeze(3)),3)
                resl = (resl*(pred_centers).unsqueeze(3)).view(batch_size,-1,3)

                pred_centers = torch.zeros(batch_size,128,128).cuda()
                for batch in range(batch_size):
                    resl[batch] = resl[batch,torch.argsort(resl[batch,:,0],descending=True)]
                    for xyscore_pixel in resl[batch,:sum_centers[batch]]:
                        if pred_centers[batch,xyscore_pixel[1].long(),xyscore_pixel[0].long()] < xyscore_pixel[2]:
                            pred_centers[batch,xyscore_pixel[1].long(),xyscore_pixel[0].long()] = xyscore_pixel[2]
#                 pred_centers = torch.tensor(pred_centers).float()

            Pr_bboxes = []
            Pr_bboxes.append(pixel_coordinates[:,:,0]+features[:,0,:,:])
            Pr_bboxes.append(pixel_coordinates[:,:,1]+features[:,1,:,:])
            Pr_bboxes.append(features[:,2,:,:])
            Pr_bboxes.append(features[:,3,:,:])
            Pr_bboxes.append(features[:,4,:,:])
            Pr_bboxes.append(pred_centers)
            Pr_bboxes = torch.stack(Pr_bboxes).permute(1,2,3,0).view(batch_size,-1,6)
            for batch in range(batch_size):
                Pr_bboxes[batch] = Pr_bboxes[batch,torch.argsort(Pr_bboxes[batch,:,5],descending=True)] #sorting
                Pr_bboxes_batch = np.array(Pr_bboxes[batch,:torch.sum(Pr_bboxes[batch,:,5]>scores_theshold)].cpu()) #selecting pixels
                if torch.sum(Pr_bboxes[batch,:,5]>scores_theshold)!=0:
                    keep = rotate_gpu_nms(np.array(Pr_bboxes_batch),nms_theshold)
                    predicted_bboxes_id = keep[:nms_saved_images_num]
                    
#                     keep = nms_rotate_cpu(np.array(Pr_bboxes_batch)[:,:5],np.array(Pr_bboxes_batch)[:,5],nms_theshold,nms_saved_images_num)
#                     predicted_bboxes_id = keep
                    
                    predicted_bboxes = [Pr_bboxes_batch[predicted_bboxes_id][np.newaxis,:]]
                else:
                    predicted_bboxes = []
                    
    end = datetime.datetime.now()
    time = (end - start).seconds+10**(-6)*(end - start).microseconds
    print('Number of img processed：',(image_count*test_epoch))
    print('Second used：',time)
    print('Fps:',(image_count*test_epoch)/time)
    
    
def imshow_label(dataset, data_index, label_show = [1,1,1,1]):
    data = dataset[data_index]
    img = data[0]
    bodies, centers, boxes, scores = data[4], data[5][0], data[6][0], data[7]
    
    img_plt = img.transpose(1,2,0)
    img_plt = (img_plt - np.min(img_plt))/(np.max(img_plt)-np.min(img_plt))
    # plt.figure(figsize=(10,10))
    plt.imshow(img_plt)
    plt.show()

    if label_show[0]:
        plt.imshow(bodies)
        plt.show()
    if label_show[1]:
        plt.imshow(centers)
        plt.show()
    if label_show[2]:
        plt.imshow(boxes)
        plt.show()
    if label_show[3]:
        plt.imshow(scores)
        plt.show()

def imshow_prediction(trained_model, dataset, data_index, pixel_coordinates, center_theshold = 0.7, scores_theshold = 0.7,num_theshold = 0.7, correct_num = 2, get_score = False, img_size = [2,2], width_drawbbox = 7,show_all = False, show_prediction = False, draw_gt = True):
    if correct_num <1:
        print('correct_num need to be more than 1')
        return
    
    data = dataset[data_index]
    ori_img = data[0].astype('int')
    img = data[1]
    bboxes = data[2]
    bboxes_count = data[3]
    id_ = data[4]
    bodies, centers, boxes, scores = data[5], data[6][0], data[7][0], data[8]
    H,W = data[9]
    o_H,o_W = 512,512
    
#     H = json_data['images'][data_index]['height']
#     W = json_data['images'][data_index]['width']
#     id_ = json_data['images'][data_index]['file_name'].split('.')[0]
    #show_image
#     img_plt = img.transpose(1,2,0)
#     img_plt = (((img_plt - np.min(img_plt))/(np.max(img_plt)-np.min(img_plt)))*255).astype('int')
    # plt.figure(figsize=(10,10))
    
    if show_all:
        plt.imshow(ori_img)
        plt.show()

    img_prediction = torch.FloatTensor(img).unsqueeze(0)
    mask,features= trained_model.cuda()(img_prediction.cuda())
    mask = mask.cpu().detach().numpy()
    if show_all:
        plt.imshow(mask[0,0,:,:])
        plt.show()
        plt.imshow(mask[0,1,:,:])
        plt.show()
    
    mask = torch.FloatTensor(mask).cuda()
    mm = (mask[0,1]>center_theshold).float()
    xx = (((features[0,5].detach().float())+pixel_coordinates[:,:,0])/4)
    yy = (((features[0,6].detach().float())+pixel_coordinates[:,:,1])/4)
    xx = (xx*(xx<128).float()*(xx>0).float()* mm).long()
    yy = (yy*(yy<128).float()*(yy>0).float()* mm).long()
    rr = torch.zeros(128,128)
    rr[yy,xx] +=1
    rr[0,0]=0
    if show_all:
        plt.imshow(rr)
        plt.show()
    if 1-get_score:
        for i in range(correct_num-1):
            mm = (rr==1).float().cuda()
            xx = (((features[0,5].detach().float())+pixel_coordinates[:,:,0])/4)
            yy = (((features[0,6].detach().float())+pixel_coordinates[:,:,1])/4)
            xx = (xx*(xx<128).float()*(xx>0).float()* mm).long()
            yy = (yy*(yy<128).float()*(yy>0).float()* mm).long()
            rr = torch.zeros(128,128)
            rr[yy,xx] = 1
            rr[0,0]=0
            if show_all:
                plt.imshow(rr)
                plt.show()
    else:
        for i in range(correct_num-1):
            if i ==0 :
                score_rr = mask[0,0]
            else:
                score_rr = rr
            ss = torch.sum((rr>0).float()).int()
            xx = (((features[0,5].round().detach().float())+pixel_coordinates[:,:,0])/4)
            yy = (((features[0,6].round().detach().float())+pixel_coordinates[:,:,1])/4)
            resl = torch.cat((xx.unsqueeze(2),yy.unsqueeze(2),score_rr.unsqueeze(2)),2)
            mass = (resl[:,:,:2]<128).float()*(resl[:,:,:2]>0).float()
            resl = (resl*(mass[:,:,0]*mass[:,:,1]*(rr>0).float().cuda()).unsqueeze(2)).view(-1,3)
            resl = resl[torch.argsort(resl[:,0],descending=True)]
            rr = np.zeros((128,128))
            for l in resl[:ss]:
                if rr[l[1].long(),l[0].long()] < l[2]:
                    rr[l[1].long(),l[0].long()] = l[2]
            rr = torch.tensor(rr).cuda().float()
            if show_all:
                plt.imshow(rr.cpu())
                plt.show()
    
    f = []
    f.append(pixel_coordinates[:,:,0]+features[0,0,:,:])
    f.append(pixel_coordinates[:,:,1]+features[0,1,:,:])
    f.append(features[0,2,:,:])
    f.append(features[0,3,:,:])
    f.append(features[0,4,:,:])
    f.append(rr.cuda())
    f = torch.stack(f).permute(1,2,0).view(-1,6)
    ll = f[torch.argsort(f[:,5])]
    if torch.sum(ll[:,5]>scores_theshold)>0:
        ll = ll[-torch.sum(ll[:,5]>scores_theshold):]
        boxes = ll[:,:6].detach().cpu().numpy()
        keep = rotate_gpu_nms(np.array(ll.detach().cpu()),num_theshold)
        result = keep[:20]
    else :
        result = []

    img_draw = ori_img
    print("The number of detected bounding boxes:",len(result))
    for i in range(len(result)):
        bb = np.zeros((5))
        prew = boxes[result[i]][2]
        preh = boxes[result[i]][3]
        prex = boxes[result[i]][0]
        prey = boxes[result[i]][1]
        prest = boxes[result[i]][4]
        prescore = boxes[result[i]][5]
        print('predicted bbox:','x:',round(prex, 2),'y:',round(prey, 2),'w:',round(prew, 2),'h:',round(preh, 2),'theta:',round(prest, 2), 'scores',round(prescore, 2))
        bb[:] = prex,prey,prew,preh,prest
        gg = bb[np.newaxis,:]
        
        gg = resize_bbox(gg, (o_H, o_W), (H, W))
        
        img_draw = draw_bbox(img_draw,gg,str(round(prescore, 2)),width_drawbbox,'yellow').get_bbox_img()

    boxv = bboxes[:bboxes_count][:,:5]
    if draw_gt:
        print("The number of real bounding boxes:",bboxes_count)
        for i in range(bboxes_count):
            boxvi = boxv[i]
            bb = np.zeros((5))
            prew = boxvi[2]
            preh = boxvi[3]
            prex = boxvi[0]
            prey = boxvi[1]
            prest = boxvi[4]
            print('Gt bbox:','x:',round(prex, 2),'y:',round(prey, 2),'w:',round(prew, 2),'h:',round(preh, 2),'theta:',round(prest,2),'scores',round(1.00, 2))
            bb[:] = prex,prey,prew,preh,prest
            gg = bb[np.newaxis,:]
            img_draw = draw_bbox(img_draw,gg,'',width_drawbbox,'green').get_bbox_img()
    if show_prediction:
        plt.figure(figsize=img_size)
        plt.imshow(img_draw)
        plt.show()

    plt.imsave('predicted_results/'+id_+'.jpg', img_draw.astype(np.uint8))

#get width and height
def real_Validate_FARN(model, epoch, tes_Dataloader, pixel_coordinates, Log_path, bestmap_07, bestmap_12, fliter_theshold = 0.5, scores_theshold = 0, nms_theshold = 0.7, nms_saved_images_num = 20, plot_PR = False):
    '''♦♦♦♦♦♦♦♦♦♦Val♦♦♦♦♦♦♦♦♦♦'''
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for i,(ori_img, img,bboxes,bboxes_count,_,bodies, centers, boxes, scores, (H,W)) in enumerate(tqdm(tes_Dataloader,desc='val')):
        img,bboxes,bboxes_count,bodies, centers, boxes, scores = img.cuda(),bboxes.cuda(),bboxes_count.cuda(),bodies.cuda(), centers.cuda(), boxes.cuda(), scores.cuda()
        H,W = float(H),float(W)
        o_H,o_W = 512,512
        
        mask,features = model(img)
        mm = (mask[0,1].detach()>fliter_theshold).float()
        pred_x_fea = (((features[0,5].round().detach().float())+pixel_coordinates[:,:,0])/4)
        pred_y_fea = (((features[0,6].round().detach().float())+pixel_coordinates[:,:,1])/4)
        pred_x_fea = (pred_x_fea*(pred_x_fea<128).float()*(pred_x_fea>0).float()* mm).long()
        pred_y_fea = (pred_y_fea*(pred_y_fea<128).float()*(pred_y_fea>0).float()* mm).long()
        pred_centers = torch.zeros(128,128).cuda()
        pred_centers[pred_y_fea,pred_x_fea] = 1
        pred_centers[0,0]=0

        sum_centers = torch.sum(pred_centers).int()
        pred_x_fea = (((features[0,5].round().detach().float())+pixel_coordinates[:,:,0])/4)
        pred_y_fea = (((features[0,6].round().detach().float())+pixel_coordinates[:,:,1])/4)
        resl = torch.cat((pred_x_fea.unsqueeze(2),pred_y_fea.unsqueeze(2),mask[0,0].detach().unsqueeze(2)),2)
        remove_0_128 = (resl[:,:,:2]<128).float()*(resl[:,:,:2]>0).float()
        resl = (resl*(remove_0_128[:,:,0]*remove_0_128[:,:,1]*pred_centers).unsqueeze(2)).view(-1,3)
        resl = resl[torch.argsort(resl[:,0],descending=True)]

        pred_centers = np.zeros((128,128))
        for xyscore_pixel in resl[:sum_centers]:
            if pred_centers[xyscore_pixel[1].long(),xyscore_pixel[0].long()] < xyscore_pixel[2]:
                pred_centers[xyscore_pixel[1].long(),xyscore_pixel[0].long()] = xyscore_pixel[2]
        pred_centers = torch.tensor(pred_centers).cuda().float()

        Pr_bboxes = []
        Pr_bboxes.append(pixel_coordinates[:,:,0]+features[0,0,:,:])
        Pr_bboxes.append(pixel_coordinates[:,:,1]+features[0,1,:,:])
        Pr_bboxes.append(features[0,2,:,:])
        Pr_bboxes.append(features[0,3,:,:])
        Pr_bboxes.append(features[0,4,:,:])
        Pr_bboxes.append(pred_centers)
        Pr_bboxes = torch.stack(Pr_bboxes).permute(1,2,0).view(-1,6)
        Pr_bboxes = Pr_bboxes[torch.argsort(Pr_bboxes[:,5])] #sorting
        Pr_bboxes = Pr_bboxes[-torch.sum(Pr_bboxes[:,5]>scores_theshold):] #selecting pixels
        Gt_bboxes = np.array(bboxes[0,:bboxes_count,:5].cpu().unsqueeze(0))

#         H = json_data['images'][i]['height']
#         W = json_data['images'][i]['width']

        Pr_bboxes = resize_bbox(np.array(Pr_bboxes.detach().cpu()), (o_H, o_W), (H, W))
        
        if np.sum(Pr_bboxes[:,5]>0)!=0:
            keep = rotate_gpu_nms(np.array(Pr_bboxes),nms_theshold)
            predicted_all_bboxes = Pr_bboxes[:,:6]
            predicted_bboxes_id = keep[:nms_saved_images_num]
            
            predicted_bboxes = predicted_all_bboxes[np.newaxis,predicted_bboxes_id]

            pred_bboxes_ =list(predicted_bboxes[:,:,:5])
            pred_labels_ = list(np.zeros((1,len(predicted_bboxes[0]))))
            pred_scores_ = list(predicted_bboxes[:,:,5])
        else:
            preiou = np.zeros((0,0,0))
            pred_bboxes_ =list(np.zeros((1,0,5)))
            pred_labels_ = list(np.zeros((1,0)))
            pred_scores_ = list(np.zeros((1,0)))

        gt_bboxes += list(Gt_bboxes[:,:,:5])
        gt_labels += list(np.zeros((1,len(Gt_bboxes[0]))))
        gt_difficults += list(np.zeros((1,len(Gt_bboxes[0]))))
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    #MAP VOC 07
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True,iou_thresh = 0.5)
    MAP_07 = result[0]['map']
    
    #MAP VOC 12
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=False,iou_thresh = 0.5)
    MAP_12 = result[0]['map']

    print(('Epoch:'+str(epoch)+'   07_MAP:'+str(MAP_07)+'   12_MAP:'+str(MAP_12)+'   Best07_Map:'+str(bestmap_07)+'   Best12_Map:'+str(bestmap_12)+'\n'))
    f = open(Log_path,'a')
    f.write('Epoch:'+str(epoch)+'   07_MAP:'+str(MAP_07)+'   12_MAP:'+str(MAP_12)+'   Best07_Map:'+str(bestmap_07)+'   Best12_Map:'+str(bestmap_12)+'\n')
    f.close()

    if plot_PR:
        PR = np.zeros((len(list(result[1][0])),2))
        PR[:,0] = np.array(list(result[1][0]))
        PR[:,1] = np.array(list(result[2][0]))
        PR = PR[PR[:,1].argsort()] 
        for i in tqdm(range(len(list(result[1][0])))):
            for ii in range(i):
                if PR[ii,0]<PR[i,0]:
                    PR[ii,0]=PR[i,0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlim=[0, 1], ylim=[0, 1], title='PR',
               ylabel='Percision', xlabel='Recall')
        plot1 = plt.plot(PR[:,1], PR[:,0], 'r',label='original values')
        plt.show()
    return MAP_07, MAP_12

def resize_bbox(bboxes, in_size, out_size):
    count = bboxes.shape[0]
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


