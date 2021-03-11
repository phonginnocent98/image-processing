#!/usr/bin/env python
#encoding: utf8
import numpy as np
import cv2


# main function
image = cv2.imread('redphong.jpg') # ファイル読み込み
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 画像をHSVに変換

# HSV extract 
# define range of red color in HSV
lower = np.array([0, 100, 100])
upper = np.array([5, 255, 255])


# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower, upper)


red = cv2.bitwise_and(image, image, mask=mask1) # 元画像とマスクを合成
gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

# 2値化
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# ラベリング処理
label = cv2.connectedComponentsWithStats(gray)

# Extract blob information
n = label[0] - 1   #　ブロブの数
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)
# ブロブ面積最大のインデックス
max_index = np.argmax(data[:,4])
# display information from max blob
center = center[max_index]
x1 = data[:,0][max_index]
y1 = data[:,1][max_index]
x2 = x1 + data[:,2][max_index]  
y2 = y1 + data[:,3][max_index]  
a = data[:,4][max_index]       

org = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0),3)
org = cv2.drawMarker(result2, (b[0], b[1]), (0, 0, 255),markerType=cv2.MARKER_SQUARE, thickness=10)  
cv2.imshow('frame',result1)
cv2.imshow('redphong',org)
cv2.imwrite('reddetectedphong1.jpg',result1)
cv2.waitKey(0)

cv2.destroyAllWindows()
