import numpy as np

# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[730.1414, 0.6085, 622.4072],
                                         [0, 729.6000, 313.2864],
                                         [0, 0, 1]
                                         ])
        # 右相机内参
        self.cam_matrix_right = np.array([[731.4855, 0.7565, ],
                                          [0, 704.9808, 312.3801],
                                          [0, 0, 1]
                                          ])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0728, -0.0817, 0, 0, 0]])
        self.distortion_r = np.array([[0.0731, -0.1012, 0, 0, 0]])

        # 旋转矩阵
        self.R = np.array([[0.9999, -0.0023, -0.0148],
                           [0.0023, 1.0000, 0.0013],
                           [0.0148, -0.0013, 0.9999]
                           ])

        # 平移矩阵
        self.T = np.array([[-5.8368], [-0.0155], [-0.0594]])

        # 焦距
        self.focal_length = 859.367  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = 5.8368  # 单位：mm， 为平移向量的第一个参数（取绝对值）




