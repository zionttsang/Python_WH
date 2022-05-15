# This class is for slef-made img related function use.
import sys
sys.path.append('C:/0_tsa/Code/Python/WH')

from pymouse import *
from pykeyboard import PyKeyboard

import gdi_capture

import random
import time
import os
import time
# from PIL import ImageGrab
# from PIL import Image
# from aip import AipOcr

import winsound
import pyautogui as pag
import cv2 as cv
# import pytesseract
import numpy as np
from matplotlib import pyplot as plt

class ImgFuction:

    def __init__(self, window_name = 'MapleStory.exe') -> None:
        self.hwnd = gdi_capture.find_window_from_executable_name("MapleStory.exe")
        # self.hwnd = gdi_capture.find_window_from_executable_name("xyqsvc.exe")

    def alert(duration = 3, times = 1):
        for t in range(times):
            d = int(duration*1000)  # milliseconds
            freq = 440  # Hz
            winsound.Beep(freq, d)
            time.sleep(0.5)

    def locate_on_img(bigImgPath, smallImgPath):
        #  match template
        img = cv.imread(bigImgPath,0)
        img2 = img.copy()
        template = cv.imread(smallImgPath,0)
        w, h = template.shape[::-1]
        
        try:
            res = cv.matchTemplate(img,template,cv.TM_SQDIFF)
            if res.any():
                return True
        except cv.error as e: 
            print ("No match was found")
            False
        '''
        # 列表中所有的6种比较方法
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                    'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            # 应用模板匹配
            res = cv.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img,top_left, bottom_right, 255, 2)
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()
        '''

    def locatesOnImg(self, bigImgPath, smallImgPath, threshold=0.98):
        print('locate attr: ', smallImgPath)
        img_rgb = cv.imread(bigImgPath)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        template = cv.imread(smallImgPath,0)
        w, h = template.shape[::-1]
        
        # threshold = 0.98
        res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
        res[res<threshold] = 0 # ??? uncertain.
        matchItem = 0
        if res.any()> threshold:
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                matchItem +=1
                print('match item:', pt)
            # '''
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            cv.imwrite('res.png',img_rgb)
            cv.imshow('res.png',img_rgb)
            cv.waitKey(0)
            # '''
        else:
            print ("No match was found")
            # return matchItem
            return pt
        
        # return matchItem
        return pt

    def getScreenImg(self, strBP = 'ss.png'):
        # strBP = 'ss.png'
        scr_img = pag.screenshot(strBP)
        # pag.alert()
        
        print ('scr_img: ', scr_img)
        scr_img = np.array(scr_img)
        scr_img = cv.cvtColor(scr_img, cv.COLOR_RGB2BGR)
        
        cv.imshow('vision', scr_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        return strBP
        
    def getAppImgByGui(self):
        with gdi_capture.CaptureWindow(self.hwnd) as img:
            locations = []
            if img is None:
                print("MapleStory.exe was not found.")
            else:
                scr_img = np.array(img)
                # scr_img = cv.cvtColor(scr_img, cv.COLOR_RGB2BGR)
                
                cv.imshow('vision', scr_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
            
if __name__ == '__main__':
    print('no compile error')
    
    imgf = ImgFuction()
    imgf.getScreenImg()
    