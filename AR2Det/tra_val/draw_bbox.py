import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont 
class draw_bbox:
    def __init__(self,img,bbox,label,width = 7,ffill='red'):
        self.img = img
        self.bbox = bbox
        self.label = label
        self.width = width
        self.ffill = ffill
    def __len__(self,):
        return len(self.bbox)
    def __rotate__(self,x,y,width,height,st):
        cosA = math.cos(st)
        sinA = math.sin(st)
        x1=x-0.5*width
        y1=y-0.5*height
        x0=x+0.5*width
        y0=y1
        x2=x1
        y2=y+0.5*height
        x3=x0
        y3=y2
        x0n= (x0 -x)*cosA -(y0 - y)*sinA + x
        y0n = (x0-x)*sinA + (y0 - y)*cosA + y
        x1n= (x1 -x)*cosA -(y1 - y)*sinA + x
        y1n = (x1-x)*sinA + (y1 - y)*cosA + y
        x2n= (x2 -x)*cosA -(y2 - y)*sinA + x
        y2n = (x2-x)*sinA + (y2 - y)*cosA + y
        x3n= (x3 -x)*cosA -(y3 - y)*sinA + x
        y3n = (x3-x)*sinA + (y3 - y)*cosA + y
        return x0n,y0n,x1n,y1n,x2n,y2n,x3n,y3n
    def get_bbox_img(self):
        self.img = Image.fromarray(self.img.astype('uint8')).convert('RGB')
        draw = ImageDraw.Draw(self.img)
        font_ = ImageFont.truetype('arialuni.ttf',20)
        for i in range(self.__len__()):
            x0n,y0n,x1n,y1n,x2n,y2n,x3n,y3n = self.__rotate__(self.bbox[i,0],self.bbox[i,1],
                                                         self.bbox[i,2],self.bbox[i,3],self.bbox[i,4])
            rand_color = int(np.random.rand(1)*255)
            draw.line([(x0n,y0n),(x1n,y1n),(x2n,y2n),(x3n,y3n),(x0n,y0n)],width=self.width,fill=self.ffill)
            draw.text((x1n,y1n),str(self.label),font = font_,ffill=self.ffill)
        self.img = np.array(self.img)
        return self.img