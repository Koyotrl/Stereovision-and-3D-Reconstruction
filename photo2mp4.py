import cv2
import os

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式

# cap_fps是帧率，可以根据随意设置
cap_fps = 20

# 注意！！！
# size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），但是此处的size要传的是（width，height），这里一定要注意注意不然结果会打不开，比如通过img.shape得到常用的图片尺寸
size = （1920,1080）

# 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
# video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
video = cv2.VideoWriter('result.mp4', fourcc, cap_fps, size)

# 这里直接读取py文件所在目录下的pics目录所有图片。
path = './photo/'
file_lst = os.listdir(path)
for filename in file_lst:
    img = cv2.imread(path + filename)
    video.write(img)
video.release()
