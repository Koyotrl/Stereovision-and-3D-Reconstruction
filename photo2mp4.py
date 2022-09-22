# import cv2
# import os
#
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
#
# # cap_fps是帧率，可以根据随意设置
# cap_fps = 3
#
# # 注意！！！
# # size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），但是此处的size要传的是（width，height），这里一定要注意注意不然结果会打不开，比如通过img.shape得到常用的图片尺寸
# size = (1280,720)
#
# # 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
# # video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
# video = cv2.VideoWriter('result.mp4', fourcc, cap_fps, size)
#
# for i in range(1, 53):
#     # 这里直接读取py文件所在目录下的pics目录所有图片。
#     path = ('/home/eaibot71/test1/test_depth/depth/%sdepth%d.png' % (string, i))
#     file_lst = os.listdir(path)
#     for filename in file_lst:
#     img = cv2.imread(path + filename)
#     video.write(img)
#     video.release()
# print("Finished All!!!")



import cv2
import os

img = cv2.imread('/home/eaibot71/test1/test_depth/depth/redepth1.png',1)
imgInfo = img.shape
size = (imgInfo[1],imgInfo[0])
print(size)

fourcc=cv2.VideoWriter_fourcc(*'mp4v')

videoWrite = cv2.VideoWriter('/home/eaibot71/test1/result1.mp4',fourcc,6,size)# 写入对象：1.fileName  2.-1：表示选择合适的编码器  3.视频的帧率  4.视频的size
for i in range(1,93):
    fileName = '/home/eaibot71/test1/test_depth/depth_wls/redepth'+ str(i) + '.png'
    img = cv2.imread(fileName)
    videoWrite.write(img)# 写入方法  1.编码之前的图片数据
print('done!')


# import numpy as np
# import cv2
# size = (1280,720)
# fourcc=cv2.VideoWriter_fourcc(*'mp4v')
# videoWrite = cv2.VideoWriter('/home/eaibot71/test1/result.mp4',-1,5,size)# 写入对象：1.fileName  2.-1：表示选择合适的编码器  3.视频的帧率  4.视频的size
# img_array=[]
# for filename in[r'/home/eaibot71/test1/test_depth/depth/redpth{}.png'.format(i)]:
#     img = cv2.imread(filename)
#     if img is None:
#         print(filename + "is error!")
#         continue
#     img_array.append(img)
#     for i in range(53):
#         videoWrite.write(img_array[i])
#     videoWrite.release()
#     print('end!')
