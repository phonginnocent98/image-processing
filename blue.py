#!/usr/bin/env python
#encoding: utf8
import rospy, cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Redtracking():
    def __init__(self):
        sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.get_image)
        self.pub = rospy.Publisher("blue", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_org = None

    def monitor(self,org):  
        hsv = cv2.cvtColor(org, cv2.COLOR_BGR2HSV) # 画像をHSVに変換                         
        hsvLower_blue = np.array([110,50,50])      # 抽出する色の下限
        hsvUpper_blue = np.array([130,255,255])    # 抽出する色の上限
        hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
        org1 = cv2.bitwise_and(org, org, mask=hsv_mask)
    # グレースケール変換
        gray = cv2.cvtColor(org1, cv2.COLOR_BGR2GRAY)

    # 2値化
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # ラベリング処理
        label = cv2.connectedComponentsWithStats(gray)

    # ブロブ情報を項目別に抽出
        n = label[0] - 1   #　ブロブの数
        data = np.delete(label[2], 0, 0)
        center = np.delete(label[3], 0, 0)  

    # ブロブ面積最大のインデックス
        try:
            max_index = np.argmax(data[:,4])

    # 面積最大ブロブの各種情報を表示
            global center_max
            center_max = center[max_index]
            x1 = data[:,0][max_index]
            y1 = data[:,1][max_index]
            x2 = x1 + data[:,2][max_index]  # x1 + 幅
            y2 = y1 + data[:,3][max_index]  # y2 + 高さ
            a = data[:,4][max_index]        # 面積
            org = cv2.rectangle(org1, (x1, y1), (x2, y2), (0, 0, 255))
        except ValueError:
            pass
        self.pub.publish(self.bridge.cv2_to_imgmsg(org, "bgr8"))
        
   
    def get_image(self,img):
        try:
            self.image_org = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def detect(self):
        if self.image_org is None:
            return None
    
        org = self.image_org
    
        self.monitor(org)   
        return center_max
       
if __name__ == '__main__':
    rospy.init_node('redtracking')
    fd = Redtracking()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.loginfo(fd.detect())
        rate.sleep()
