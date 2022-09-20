import cv2
import numpy as np


def blur_demo(image):#均值模糊 : 去随机噪声有很好的去噪效果
    #（1, 15）是垂直方向模糊，（15， 1）是水平方向模糊
    dst = cv2.blur(image, (1, 15))
    cv2.imshow("avg_blur_demo", dst)

def median_blur_demo(image):    # 中值模糊  对椒盐噪声有很好的去燥效果
    dst = cv2.medianBlur(image, 5)
    cv2.imshow("median_blur_demo", dst)

def custom_blur_demo(image):

    kernel = np.ones([5, 5], np.float32)/25 #用户自定义模糊下面除以25是防止数值溢出
    dst = cv2.filter2D(image, -1, kernel)
    cv2.imshow("custom_blur_demo", dst)

src = cv2.imread("/home/eaibot71/test1/test_depth/depth/redepth1.png")
img = cv2.resize(src,None,fx=0.8,fy=0.8,interpolation=cv2.INTER_CUBIC)
cv2.imshow('input_image', img)

blur_demo(img)
median_blur_demo(img)
custom_blur_demo(img)

cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import math
# import os
# import numpy as np
#
# def mean_filter(src,kernel_size,epoch):#均值滤波
#     kernel_height,kernel_weight=kernel_size
#     dst=src
#     pad_num=int(np.floor(kernel_weight/2))#计算需要加的边
#     pad_src=np.pad(src,((pad_num,pad_num),(pad_num,pad_num)),'constant')#加边
#     for z in range(epoch):#迭代
#         for i in range(pad_num,pad_src.shape[0]-pad_num):#滤波
#             for j in range(pad_num,pad_src.shape[1]-pad_num):
#                 sum_temp=sum(pad_src[i-pad_num:i+pad_num+1,j-pad_num:j+pad_num+1]).tolist()#求和
#                 #print(sum_temp)
#                 sum_temp=sum(sum_temp)
#                 #print(sum_temp)
#                 temp=sum_temp/(kernel_weight*kernel_height)#求均值
#                 dst[i-pad_num,j-pad_num]=temp#更新像素
#         src=dst#更新原图像
#         pad_src = np.pad(src, ((pad_num, pad_num), (pad_num, pad_num)), 'constant')
#     dst = dst.astype(np.uint8)
#     return dst
#
# def guass_kernel(kernel_size,sigma):#创建高斯kernel
#     kernel_height,kernel_weight=kernel_size
#     kernel=np.zeros(kernel_size)#创建高斯kernel
#     for i in range(kernel_height):
#         for j in range(kernel_weight):
#             r=math.pow((i-(kernel_height-1)/2),2)#计算x
#             c=math.pow((j-(kernel_weight-1)/2),2)#计算y
#             kernel[i,j]=(1/(2*math.pi*math.pow(sigma,2)))*math.exp(-(math.pow(r,2)+math.pow(c,2))/(2*math.pow(sigma,2)))#按照高斯公式计算每一个kernel的值
#     sum_temp=sum(kernel)#求和
#     kernel=kernel/sum_temp#最终的高斯kernel
#     return kernel
#
# def sliding_window(src,kernel,stride):#用于单通道滑窗
#     dst = src
#     kernel_height,kernel_weight=kernel.shape#获取kernel的高度和宽度
#     exnum=int(np.floor(kernel_height/2))#填充的维数,floor函数向下取整
#     img_pad=np.pad(src,((exnum,exnum),(exnum,exnum)),'constant')#给图像加边,方便滑窗
#     for i in range(exnum,img_pad.shape[0]-exnum,stride):
#         for j in range(exnum,img_pad.shape[1]-exnum,stride):
#             temp=(img_pad[i-exnum:i+exnum+1,j-exnum:j+exnum+1]*kernel)/(kernel_height*kernel_weight)#截取n*n的mat与kenel对应相乘
#             sum_temp=sum(temp).tolist()#相加
#             sum_temp=sum(sum_temp)
#             img_pad[i,j]=sum_temp#更新像素
#             if img_pad[i,j]<0:#截断操作
#                 img_pad[i,j]=abs(img_pad[i,j])
#             if img_pad[i,j]>255:
#                 img_pad[i,j]=255
#             dst[i-exnum,j-exnum]=img_pad[i,j]
#     dst=dst.astype(np.uint8)
#     return dst
#
#
# # #均值滤波
# # img=cv2.imread("/home/eaibot71/test1/test_depth/depth/redepth1.png",0)
# # dst=mean_filter(img,(1280,720),5)
# # print(dst)
# # cv2.imshow("dst",dst)
# # cv2.waitKey(0)
# #中值滤波
# # img=cv2.imread("1.png",0)
# # dst=mean_filter(img,(3,3),5)
# # print(dst)
# # cv2.imshow("dst",dst)
# # cv2.waitKey(0)
# #高斯滤波
# kernel=guass_kernel((1280,720),1.5)
# print(kernel)
# img=cv2.imread("/home/eaibot71/test1/test_depth/depth/redepth1.png",0)
# dst=sliding_window(img,kernel,1)
# cv2.imshow("dst",dst)
# cv2.waitKey(0)
# #sobel
# # sobel_kernel_horizontal,sobel_kernel_vertical=sobel_kernel()
# # img=cv2.imread("1.png",0)
# # dst_horizontal=sliding_window(img,sobel_kernel_horizontal,1)
# # dst_vertical=sliding_window(img,sobel_kernel_vertical,1)
# # dst=dst_horizontal+dst_vertical
# # cv2.imshow("dst_horizontal",dst_horizontal)
# # cv2.imshow("dst_vertical",dst_vertical)
# # cv2.imshow("dst",dst)
# # cv2.waitKey(0)


