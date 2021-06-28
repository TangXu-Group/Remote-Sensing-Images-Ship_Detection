import torch
def logfunction(f,b):
    return (f>=b).float().cuda()*torch.log(f/b)+(f<b).float().cuda()*torch.log(b/f)

def smoothL1(tx,ty,tw,th,tst):
    tt = torch.cat((tx.unsqueeze(1),ty.unsqueeze(1),tw.unsqueeze(1),th.unsqueeze(1),tst.unsqueeze(1)),dim =1)
    tt = torch.sum(((torch.abs(tt)>=1).float()*(torch.abs(tt)-0.5)+(((torch.abs(tt)<1).float()*tt)**2)*0.5),dim = 1)
    return tt 

def smoothL1_2(tx,ty):
    tt = torch.cat((tx.unsqueeze(1),ty.unsqueeze(1)),dim =1)
    tt = torch.sum(((torch.abs(tt)>=1).float()*(torch.abs(tt)-0.5)+(((torch.abs(tt)<1).float()*tt)**2)*0.5),dim = 1)
    return tt 