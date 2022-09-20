import cv2
import  numpy as np

#图像的腐蚀膨胀
img = cv2.imread("/home/eaibot71/test1/test_depth/depth/redepth1.png")
kernel1 = np.ones((6,6),np.uint8)#卷积核
kernel2 = np.ones((6,6),np.uint8)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
imgCanny = cv2.Canny(imgBlur,50,100)
#interations 迭代运算次数

imgDialation = cv2.dilate(imgGray,kernel1,iterations=)  #膨胀
imgEroded = cv2.erode(imgDialation,kernel2,iterations=1)  #腐蚀

cv2.imshow("imgBefore",img)
cv2.imshow("imgAfter1",imgDialation)
cv2.imshow("imgAfter2",imgEroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
