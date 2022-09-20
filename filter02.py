import cv2
import numpy as np
import pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
profile = pipe.start(cfg)

try:
    while True:
        frame = pipe.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        print('capture success')
        if cv2.waitKey(10)&0xff == ord('q'):
            break

finally:
    pipe.stop()
