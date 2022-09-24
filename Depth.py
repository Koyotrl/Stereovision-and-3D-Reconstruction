# -*- coding: utf-8 -*-
import cv2
import numpy as np
import stereoconfig  # 导入相机标定的参数
import pcl
import pcl.pcl_visualization


# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):  # 判断为三维数组
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,                        #最小视差
              'numDisparities': 64,                     #视差的搜索范围，16的整数倍
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,  #值越大，视差越平滑，相邻像素视差+/-1的惩罚系数
              'P2': 32 * img_channels * blockSize ** 2, #同上，相邻像素视差变化值>1的惩罚系数
              'disp12MaxDiff': 1,                       #左右一致性检测中最大容许误差值
              'preFilterCap': 63,                       #映射滤波器大小，默认15
              'uniquenessRatio': 15,                    #唯一检测性参数，匹配区分度不够，则误匹配(5-15)
              'speckleWindowSize': 100,                 #视差连通区域像素点个数的大小（噪声点）(50-200)或用0禁用斑点过滤
              'speckleRange': 1,                        #认为不连通(1-2)
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right

# 视差计算+wls滤波
def stereoMatchSGBM2(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,                        #最小视差
              'numDisparities': 64,                     #视差的搜索范围，16的整数倍
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,  #值越大，视差越平滑，相邻像素视差+/-1的惩罚系数
              'P2': 32 * img_channels * blockSize ** 2, #同上，相邻像素视差变化值>1的惩罚系数
              'disp12MaxDiff': 1,                       #左右一致性检测中最大容许误差值
              'preFilterCap': 63,                       #映射滤波器大小，默认15
              'uniquenessRatio': 15,                    #唯一检测性参数，匹配区分度不够，则误匹配(5-15)
              'speckleWindowSize': 100,                 #视差连通区域像素点个数的大小（噪声点）(50-200)或用0禁用斑点过滤
              'speckleRange': 1,                        #认为不连通(1-2)
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    # sigmaColor典型范围值为0.8-2.0
    wls_filter.setLambda(8000.)
    wls_filter.setSigmaColor(2.0)
    wls_filter.setLRCthresh(24)
    wls_filter.setDepthDiscontinuityRadius(3)
    filtered_disp = wls_filter.filter(trueDisp_left, left_image, disparity_map_right=trueDisp_right)
    disp22 = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disp22


# wls_filter
def wls_filter(left_image, right_image):

    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3

    paraml = {
    'preFilterCap': 63,     #映射滤波器大小，默认15
    "minDisparity" : 0,    #最小视差
    "numDisparities" : 64,    #视差的搜索范围，16的整数倍
    "blockSize" : blockSize,
    "uniquenessRatio" : 15,     #唯一检测性参数，匹配区分度不够，则误匹配(5-15)
    "speckleWindowSize" : 100,      #视差连通区域像素点个数的大小（噪声点）(50-200)或用0禁用斑点过滤
    "speckleRange" : 1,             #认为不连通(1-2)
    "disp12MaxDiff" : 1,        #左右一致性检测中最大容许误差值
    "P1" : 8 * img_channels * blockSize** 2,   #值越大，视差越平滑，相邻像素视差+/-1的惩罚系数
    "P2" : 32 * img_channels * blockSize** 2,  #同上，相邻像素视差变化值>1的惩罚系数
    # 'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
    }

    ## 开始计算深度图
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    left_disp = left_matcher.compute(left_image, right_image)
    right_disp = right_matcher.compute(right_image, left_image)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    # sigmaColor典型范围值为0.8-2.0
    wls_filter.setLambda(4000.)
    wls_filter.setSigmaColor(0.8)
    wls_filter.setLRCthresh(24)
    wls_filter.setDepthDiscontinuityRadius(1)

    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)

    disp2 = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disp2


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = -pointcloud[:, 1]
    Z = -pointcloud[:, 2]

    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1


# 点云显示
def view_cloud(pointcloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointcloud)

    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass


# 利用opencv函数计算深度图
def getDepthMapWithQ(disparityMap : np.ndarray, Q : np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)

# 根据公式计算深度图
def getDepthMapWithConfig(disparityMap : np.ndarray, config : stereoconfig.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)



def median_blur_demo(image):    # 中值模糊  对椒盐噪声有很好的去燥效果
    dst = cv2.medianBlur(image, 5)
    cv2.imshow("median_blur_demo", dst)



if __name__ == '__main__':

    for i in range(1, 125):
        #i = 1
        string = 're'
        # 读取数据集的图片
        iml = cv2.imread('/home/eaibot71/test1/photo_all/left/%sLeft%d.bmp' % (string, i))  # 左图
        imr = cv2.imread('/home/eaibot71/test1/photo_all/right/%sRight%d.bmp' % (string, i))  # 右图
        height, width = iml.shape[0:2]

        # print("width = %d \n" % width)
        # print("height = %d \n" % height)

        # 读取相机内参和外参
        config = stereoconfig.stereoCamera()

        # 立体校正
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

        # print("Print Q!")
        # print(Q)

        # 绘制等间距平行线，检查立体校正的效果
        line = draw_line(iml_rectified, imr_rectified)
        cv2.imwrite('/home/eaibot71/test1/test_depth/check/%scheck%d.png' % (string, i), line)

        # 消除畸变
        iml = undistortion(iml, config.cam_matrix_left, config.distortion_l)
        imr = undistortion(imr, config.cam_matrix_right, config.distortion_r)

        # 立体匹配
        iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以

        iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)

        disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, True)
        cv2.imwrite('/home/eaibot71/test1/test_depth/depth/%sdepth%d.png' % (string, i), disp)

        #wls_filter
        disp2 = wls_filter(iml, imr)
        cv2.imwrite('/home/eaibot71/test1/test_depth/depth_wls/%sdepth%d.png' % (string, i), disp2)

        disp22 = stereoMatchSGBM2(iml_, imr_, True)
        cv2.imwrite('/home/eaibot71/test1/test_depth/wls/%sdepth%d.png' % (string, i), disp22)



        #图像的腐蚀膨胀
        img = cv2.imread('/home/eaibot71/test1/test_depth/depth/%sdepth%d.png' % (string, i))
        kernel1 = np.ones((15,15),np.uint8)
        kernel2 = np.ones((7,7),np.uint8)
        kernel3 = np.ones((5,5),np.uint8)
        kernel4 = np.ones((10,10),np.uint8)

        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
        imgCanny = cv2.Canny(imgBlur,50,100)

        imgDialation = cv2.dilate(imgGray,kernel1,iterations=10)  #膨胀 闭运算
        imgEroded = cv2.erode(imgDialation,kernel2,iterations=10)  #腐蚀
        imgEroded2 = cv2.erode(imgEroded,kernel3,iterations=1)  #腐蚀 开运算
        imgDialation2 = cv2.dilate(imgEroded2,kernel4,iterations=1)  #膨胀
        cv2.imwrite('/home/eaibot71/test1/test_depth/depth_de/%sdepth%d.png' % (string, i), disp)


        # 中值模糊 将图片保存在depth_filter
        # src = cv2.imread("/home/eaibot71/test1/test_depth/depth/%sdepth%d.png" % (string, i))
        # img = cv2.resize(src,None,fx=0.8,fy=0.8,interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('/home/eaibot71/test1/test_depth/depth_filter/%sdepth%d.png' % (string, i), disp)


        # 计算像素点的3D坐标（左相机坐标系下）
        points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数

        print('image done:', i)

        # points_3d = points_3d

    # 鼠标点击事件
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (
            x, y, points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]))
            dis = ((points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] ** 2) ** 0.5) / 1000
            print('点 (%d, %d) 距离左摄像头的相对距离为 %0.3f m' % (x, y, dis))

    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数

    # 构建点云--Point_XYZRGBA格式
    pointcloud = DepthColor2Cloud(points_3d, iml)



    # 显示图片
    cv2.namedWindow("disparity", 0)
    cv2.imshow("disparity", disp)
    cv2.setMouseCallback("disparity", onMouse, 0)

    # 显示点云
    view_cloud(pointcloud)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
