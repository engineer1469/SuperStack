# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:41:20 2019

@author: Copper280z
"""
import types
import numpy as np
import cv2


class reader(object):
    def __init__(self,fname):
        self.fname=fname
        self.header=types.SimpleNamespace()
        with open(self.fname,"rb") as f:
            self.header.fileID=f.read(14).decode()
            self.header.luID=int.from_bytes(f.read(4), byteorder='little')
            self.header.colorID=int.from_bytes(f.read(4), byteorder='little')
            """
            Content:
                MONO= 0
                BAYER_RGGB= 8
                BAYER_GRBG= 9
                BAYER_GBRG= 10
                BAYER_BGGR= 11
                BAYER_CYYM= 16
                BAYER_YCMY= 17
                BAYER_YMCY= 18
                BAYER_MYYC= 19
                RGB= 100
                BGR= 101
            """
            if self.header.colorID <99:
                self.header.numPlanes = 1
            else:
                self.header.numPlanes = 3
                
            self.header.littleEndian=int.from_bytes(f.read(4), byteorder='little')
            self.header.imageWidth=int.from_bytes(f.read(4), byteorder='little')
            self.header.imageHeight=int.from_bytes(f.read(4), byteorder='little')
            self.header.PixelDepthPerPlane=int.from_bytes(f.read(4), byteorder='little')
            if self.header.PixelDepthPerPlane == 8:
                self.dtype = np.uint8
            elif self.header.PixelDepthPerPlane == 16:
                self.dtype = np.uint16
            self.header.frameCount=int.from_bytes(f.read(4), byteorder='little')
            self.header.observer=f.read(40).decode()
            self.header.instrument=f.read(40).decode()
            self.header.telescope=f.read(40).decode()
            self.header.dateTime=int.from_bytes(f.read(8), byteorder='little')
            self.imgSizeBytes = int(self.header.imageHeight*self.header.imageWidth*self.header.PixelDepthPerPlane*self.header.numPlanes/8)
            self.imgNum=0
        
    def getImg(self,imgNum=None):
        if imgNum is None:
            pass
        else:
            self.imgNum=imgNum
            
        with open(self.fname,"rb") as f:
            f.seek(int(178+self.imgNum*(self.imgSizeBytes)))
            frame = np.frombuffer(f.read(self.imgSizeBytes),dtype=self.dtype)
            
        self.imgNum+=1
        
        frame = np.reshape(frame,(self.header.imageHeight,self.header.imageWidth,self.header.numPlanes))
        return frame
print(__name__)
if __name__ == '__main__':
    fname=r'F:\2019-12-23\Capture\00_39_12.ser'
    ser = reader(fname)
    for n in range(ser.header.frameCount):
        frame = ser.getImg()
        frame=cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)
        cv2.imshow('x',frame)
        cv2.waitKey(int(1000/30))
    cv2.destroyAllWindows()
    cv2.waitKey(1)
                