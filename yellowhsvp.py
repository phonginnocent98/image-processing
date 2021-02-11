#!/usr/bin/env python
#encoding: utf8
import numpy as np
import cv2


# メイン関数
image = cv2.imread('lemon1.jpg') # ファイル読み込み
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 画像をHSVに変換

# HSVでの色抽出
# HSVで黄色の上限と下限
lower = np.array([10, 100, 100])
upper = np.array([35, 255, 255])


#マスクを作成する
mask1 = cv2.inRange(hsv, lower, upper)


red = cv2.bitwise_and(image, image, mask=mask1) # 元画像とマスクを合成
gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

# 2値化
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# ラベリング処理
label = cv2.connectedComponentsWithStats(gray)

# ブロブ情報を項目別に抽出
n = label[0] - 1   #　ブロブの数
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)
# ブロブ面積最大のインデックス
max_index = np.argmax(data[:,4])
# 面積最大ブロブの各種情報を表示
center = center[max_index]
x1 = data[:,0][max_index]
y1 = data[:,1][max_index]
x2 = x1 + data[:,2][max_index]  
y2 = y1 + data[:,3][max_index]  
a = data[:,4][max_index]       

b=[]
for i in range(2):
    x = int(format(center[i], '.0f'))
    b.append(x)
    
result1 = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0),3)
result2 = cv2.rectangle(red, (x1, y1), (x2, y2), (0, 255, 0),3)
org = cv2.drawMarker(result2, (b[0], b[1]), (0, 0, 255),markerType=cv2.MARKER_SQUARE, thickness=10)  
cv2.imshow('frame',result1)
cv2.imshow('lemondetect',org)
cv2.imwrite('lemondetect.jpg',result2)
cv2.waitKey(0)

cv2.destroyAllWindows()
