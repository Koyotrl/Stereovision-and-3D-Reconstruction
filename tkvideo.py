import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('/home/eaibot71/test1/photo_video/output.mp4',fourcc, 5.0, (width,height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
#        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   将视频转换为灰色的源
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 若按下Q键，则退出循环
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
