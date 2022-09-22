import numpy as np
import cv2

# img1 = cv2.imread(r"/Users/inbc/Desktop/zuo/Left1.bmp")
# img2 = cv2.imread(r"/Users/inbc/Desktop/you/Right1.bmp")
for i in range(1, 125):
    # imgT = cv2.imdecode(np.fromfile('./images/%d.bmp'  %i ,dtype=np.uint8), -1)
    imgT = cv2.imdecode(np.fromfile('/home/eaibot71/test1/photo_video/%d.bmp' % i, dtype=np.uint8), -1)  # 读取拍摄的左右双目照片
    #imgT = cv2.imdecode(np.fromfile('/home/eaibot71/test1/checkerboard/%d.bmp' % i, dtype=np.uint8), -1)  # 读取拍摄的左右双目照片

    # cv2.imshow("zuo", img1[300:1200, 500:2000])
    # cv2.imshow("you", img2[300:1200, 500:2000])

    # cv2.waitKey(0)

    # 设置左右照片的存储位置
    cv2.imwrite("/home/eaibot71/test1/photo_all/left/reLeft%d.bmp" % i, imgT[0:720, 0:1280])  # imgL的第一个参数是图片高度像素范围，第二个参数是图片宽度的像素范围
    cv2.imwrite("/home/eaibot71/test1/photo_all/right/reRight%d.bmp" % i, imgT[0:720, 1280:2560])
    #cv2.imwrite("/home/eaibot71/test1/checkerboard/left/reLeft%d.bmp" % i, imgT[0:720, 0:1280])  # imgL的第一个参数是图片高度像素范围，第二个参数是图片宽度的像素范围
    #cv2.imwrite("/home/eaibot71/test1/checkerboard/right/reRight%d.bmp" % i, imgT[0:720, 1280:2560])

    print("Resize images%d Fnished!" % i)

print("Finished All!!!")
