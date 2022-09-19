# 导入所需要的库
import cv2
import numpy as np

# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
        address = addr + str(num) + '.bmp'
        cv2.imwrite(address, image)

# 读取视频文件 视频文件路径
videoCapture = cv2.VideoCapture("/home/eaibot71/test1/photo_video/output.mp4")
# 通过摄像头的方式
# videoCapture=cv2.VideoCapture(1)
 
# 读帧
success, frame = videoCapture.read()
i = 0
timeF = 1
j = 0
while success:

    if (i % timeF == 0):
        j = j + 1
        save_image(frame, '/home/eaibot71/test1/photo_video/', j) #视频截成图片存放的位置
        print('save image:', i)
        i = i + 1
    success, frame = videoCapture.read()
