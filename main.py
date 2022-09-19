import cv2
import sys

# 引入库

cap = cv2.VideoCapture(0)  # 读取笔记本内置摄像头或者0号摄像头
# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 0
while True:
    ret, frame = cap.read()

    if (ret):
        cv2.namedWindow("Video01", 0)  # 创建一个名为Video01的窗口，0表示窗口大小可调
        # cv2.resizeWindow("Video01",1280,720) ##创建一个名为Video01的窗口，设置窗口大小为 1920 * 1080 与上一个设置的 0 有冲突
        cv2.imshow("Video01", frame)

        # 等待按键按下
        c = cv2.waitKey(1) & 0xff

        # r若按下w则保存一张照片
        if c == ord("w"):
            cv2.imwrite("/home/eaibot71/test1/checkerboard/all/%d.bmp" % i, frame)  # 将照片保存至棋盘格文件夹中
            #cv2.imwrite("/home/eaibot71/test1/photo_all/all/%d.bmp" % i, frame)  # 将照片保存至所有照片文件夹中
            print("Save images %d succeed!" % i)
            i += 1

        # 若按下Q键，则退出循环
        if c == ord("q"):
            break

# 随时准备按q退出
cap.release()
# 关掉所有窗口
cv2.destroyAllWindows()
