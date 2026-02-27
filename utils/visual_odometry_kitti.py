import numpy as np
from config_loader import CONFIG

class VisualOdometry:
    def __init__(self, **kwargs):
        """
        初始化视觉里程计参数
        **kwargs: 用于覆盖默认参数的关键字参数
        """
        self.ODO_IMG_TRANS_Y_RANGE = kwargs.pop(
            "ODO_IMG_TRANS_Y_RANGE",
            CONFIG.get("ODO_IMG_TRANS_Y_RANGE", slice(50, 333))
        )

        self.ODO_IMG_TRANS_X_RANGE = kwargs.pop(
            "ODO_IMG_TRANS_X_RANGE",
            CONFIG.get("ODO_IMG_TRANS_X_RANGE", slice(200, 1000))
        )

        # 用于计算水平旋转（Yaw）的图像区域（通常关注图像中部或远处特征）
        self.ODO_IMG_YAW_ROT_Y_RANGE = kwargs.pop(
            "ODO_IMG_YAW_ROT_Y_RANGE",
            CONFIG.get("ODO_IMG_YAW_ROT_Y_RANGE", slice(100, 300))
        )
        self.ODO_IMG_YAW_ROT_X_RANGE = kwargs.pop(
            "ODO_IMG_YAW_ROT_X_RANGE",
            CONFIG.get("ODO_IMG_YAW_ROT_X_RANGE", slice(350, 850))
        )

        # 用于计算垂直/高度速度的图像区域
        self.ODO_IMG_HEIGHT_V_Y_RANGE = kwargs.pop(
            "ODO_IMG_HEIGHT_V_Y_RANGE",
            CONFIG.get("ODO_IMG_HEIGHT_V_Y_RANGE", slice(100, 333))
        )
        self.ODO_IMG_HEIGHT_V_X_RANGE = kwargs.pop(
            "ODO_IMG_HEIGHT_V_X_RANGE",
            CONFIG.get("ODO_IMG_HEIGHT_V_X_RANGE", slice(200, 1000))
        )

        # --- 图像缩放尺寸 ---
        # 将裁剪后的图像缩放到统一尺寸，减少计算量并归一化输入
        self.ODO_IMG_TRANS_RESIZE_RANGE = kwargs.pop(
            "ODO_IMG_TRANS_RESIZE_RANGE",
            CONFIG.get("ODO_IMG_TRANS_RESIZE_RANGE", [CONFIG["image_Y"], CONFIG["image_X"]])
        )
        self.ODO_IMG_YAW_ROT_RESIZE_RANGE = kwargs.pop(
            "ODO_IMG_YAW_ROT_RESIZE_RANGE",
            CONFIG.get("ODO_IMG_YAW_ROT_RESIZE_RANGE", [CONFIG["image_Y"], CONFIG["image_X"]])
        )
        self.ODO_IMG_HEIGHT_V_RESIZE_RANGE = kwargs.pop(
            "ODO_IMG_HEIGHT_V_RESIZE_RANGE",
            CONFIG.get("ODO_IMG_HEIGHT_V_RESIZE_RANGE", [CONFIG["image_Y"], CONFIG["image_X"]])
        )

        # --- 速度比例系数 (Scale) ---
        # 算法计算出的是像素差异或位移值，需要乘以这些系数转换为实际的物理速度单位（如 m/s 或 deg/s）
        self.ODO_TRANS_V_SCALE = kwargs.pop(
            "ODO_TRANS_V_SCALE",
            CONFIG.get("ODO_TRANS_V_SCALE", 30)
        )  # 前进速度比例
        self.ODO_YAW_ROT_V_SCALE = kwargs.pop(
            "ODO_YAW_ROT_V_SCALE",
            CONFIG.get("ODO_YAW_ROT_V_SCALE", 1)
        )  # 旋转速度比例
        self.ODO_HEIGHT_V_SCALE = kwargs.pop(
            "ODO_HEIGHT_V_SCALE",
            CONFIG.get("ODO_HEIGHT_V_SCALE", 5)
        )  # 高度速度比例

        # --- 速度阈值 (Threshold) ---
        # 用于过滤异常值（Outliers）。如果计算出的速度变化过大，则认为计算错误，保持上一帧速度。
        self.MAX_TRANS_V_THRESHOLD = kwargs.pop(
            "MAX_TRANS_V_THRESHOLD",
            CONFIG.get("MAX_TRANS_V_THRESHOLD", 0.4)
        )
        self.MAX_YAW_ROT_V_THRESHOLD = kwargs.pop(
            "MAX_YAW_ROT_V_THRESHOLD",
            CONFIG.get("MAX_YAW_ROT_V_THRESHOLD", 4.2)
        )
        self.MAX_HEIGHT_V_THRESHOLD = kwargs.pop(
            "MAX_HEIGHT_V_THRESHOLD",
            CONFIG.get("MAX_HEIGHT_V_THRESHOLD", 0.4)
        )

        # --- 匹配搜索范围 ---
        # 在对比前后两帧时，最大允许搜索的像素偏移量。
        # 例如 HORI=30 意味着只在左右30个像素范围内寻找最佳匹配。
        self.ODO_SHIFT_MATCH_VERT = kwargs.pop(
            "ODO_SHIFT_MATCH_VERT",
            CONFIG.get("ODO_SHIFT_MATCH_VERT", 30)
        )  # 垂直方向搜索范围
        self.ODO_SHIFT_MATCH_HORI = kwargs.pop(
            "ODO_SHIFT_MATCH_HORI",
            CONFIG.get("ODO_SHIFT_MATCH_HORI", 30)
        )  # 水平方向搜索范围

        # --- 相机视场角 (FOV) ---
        # 用于将像素位移转换为角度变化
        self.FOV_HORI_DEGREE = kwargs.pop(
            "FOV_HORI_DEGREE",
            CONFIG.get("FOV_HORI_DEGREE", 109)
        )  # 水平视场角
        self.FOV_VERT_DEGREE = kwargs.pop(
            "FOV_VERT_DEGREE",
            CONFIG.get("FOV_VERT_DEGREE", 60)
        )  # 垂直视场角

        # --- 初始化状态变量 ---
        # 存储"上一帧"的图像特征（列求和或行求和后的1D数组），用于与"当前帧"对比
        # 数组大小根据Resize后的尺寸决定
        self.PREV_YAW_ROT_V_IMG_X_SUMS = np.zeros(self.ODO_IMG_TRANS_RESIZE_RANGE[1])
        self.PREV_HEIGHT_V_IMG_Y_SUMS = np.zeros(self.ODO_IMG_HEIGHT_V_RESIZE_RANGE[0])
        self.PREV_TRANS_V_IMG_X_SUMS = np.zeros(self.ODO_IMG_TRANS_RESIZE_RANGE[1] - self.ODO_SHIFT_MATCH_HORI)

        # 存储上一帧计算出的速度，用于平滑处理和异常值回退
        self.PREV_TRANS_V = 0.02
        self.PREV_YAW_ROT_V = 0.0
        self.PREV_HEIGHT_V = 0.0

        self.DEGREE_TO_RADIAN = np.pi / 180
        self.OFFSET_YAW_ROT = None
        self.OFFSET_HEIGHT_V = None

    @staticmethod
    def compare_segments(seg1, seg2, shift_length, compare_length_of_intensity):
        """
        核心算法：对比两个1D强度曲线（当前帧和上一帧），找到最佳匹配偏移量。

        原理：
        通过在一定范围（shift_length）内左右滑动seg1，计算它与seg2的重叠部分的
        绝对差值之和（SAD），找到差异最小的位置。

        参数:
        - seg1: 当前帧的1D强度分布数组
        - seg2: 上一帧的1D强度分布数组
        - shift_length: 搜索的最大像素偏移量（搜索窗口大小）
        - compare_length_of_intensity: 实际参与比较的数组长度

        返回:
        - out_minimum_offset: 最佳匹配时的偏移量（正数代表向一个方向移，负数代表反向）
        - out_minimum_difference_intensity: 最佳匹配时的最小平均差异值
        """
        minimum_difference_intensity = 1e6  # 初始化为一个很大的差异值
        minimum_offset = 0

        # 1. 搜索正向偏移 (0 到 shift_length)
        # 假设物体向一个方向移动，导致图像像素发生正向位移
        for offset in range(shift_length + 1):
            # 截取两段数组进行对比：
            # seg1 从 offset 开始，seg2 从 0 开始（相当于seg1向左移了offset与seg2对齐）
            compare_difference_segments = np.abs(seg1[offset: compare_length_of_intensity] -
                                                 seg2[: compare_length_of_intensity - offset])
            # 计算平均差异（归一化，消除长度不同带来的影响）
            sum_compare_difference_segments = sum(compare_difference_segments) / (compare_length_of_intensity - offset)

            # 更新最小值
            if sum_compare_difference_segments <= minimum_difference_intensity:
                minimum_difference_intensity = sum_compare_difference_segments
                minimum_offset = offset

        # 2. 搜索负向偏移 (1 到 shift_length)
        # 假设物体向相反方向移动
        for offset in range(1, shift_length + 1):
            # seg1 从 0 开始，seg2 从 offset 开始（相当于seg1向右移了offset与seg2对齐）
            compare_difference_segments = np.abs(seg1[: compare_length_of_intensity - offset] -
                                                 seg2[offset: compare_length_of_intensity])
            sum_compare_difference_segments = sum(compare_difference_segments) / (compare_length_of_intensity - offset)

            if sum_compare_difference_segments <= minimum_difference_intensity:
                minimum_difference_intensity = sum_compare_difference_segments
                minimum_offset = -offset  # 标记为负偏移

        return minimum_offset, minimum_difference_intensity

    def visual_odometry(self, rawImg, model=0):
        """
        简单的视觉里程计主函数。
        输入:
            rawImg: 原始图像数据
            model: 运动模式 (0:平地/普通, 1:上坡/向上, 2:下坡/向下)
        输出:
            transV: 前进线速度
            yawRotV: 旋转角速度
            heightV: 垂直/俯仰速度
        """

        # ==========================================
        # 1. 计算水平旋转角速度 (Yaw)
        # ==========================================

        # 裁剪出用于计算旋转的ROI区域
        subRawImg = rawImg[self.ODO_IMG_YAW_ROT_Y_RANGE, self.ODO_IMG_YAW_ROT_X_RANGE]
        # 缩放到标准尺寸
        subRawImg = np.float32(imresize(subRawImg, self.ODO_IMG_YAW_ROT_RESIZE_RANGE))

        # 计算每个像素代表的水平角度
        horiDegPerPixel = self.FOV_HORI_DEGREE / subRawImg.shape[1]

        # --- 降维处理 (2D -> 1D) ---
        # 将图像沿Y轴（垂直方向）求和，得到一个X轴方向的强度分布曲线。
        # 这个曲线反映了图像在水平方向上的纹理变化。
        # 如果相机水平旋转，这个曲线会左右平移。
        imgXSums = np.sum(subRawImg, axis=0)

        # 归一化：除以平均强度，消除环境光照整体变亮/变暗的影响
        avgIntensity = np.mean(imgXSums)
        imgXSums = imgXSums / avgIntensity

        # 对比当前帧(imgXSums)与上一帧(self.PREV_YAW_ROT_V_IMG_X_SUMS)
        minOffsetYawRot, minDiffIntensityRot = self.compare_segments(
            imgXSums, self.PREV_YAW_ROT_V_IMG_X_SUMS,
            self.ODO_SHIFT_MATCH_HORI, imgXSums.shape[0]
        )

        self.OFFSET_YAW_ROT = minOffsetYawRot

        # 计算旋转角速度：偏移像素数 * 单像素角度 * 比例系数
        yawRotV = self.ODO_YAW_ROT_V_SCALE * minOffsetYawRot * horiDegPerPixel  # 单位: deg

        # 阈值过滤：如果速度突变太大，认为是噪声，沿用上一帧速度
        if abs(yawRotV) > self.MAX_YAW_ROT_V_THRESHOLD:
            yawRotV = self.PREV_YAW_ROT_V
        else:
            self.PREV_YAW_ROT_V = yawRotV

        # 更新上一帧的数据
        self.PREV_YAW_ROT_V_IMG_X_SUMS = imgXSums
        self.PREV_TRANS_V_IMG_X_SUMS = imgXSums

        # ==========================================
        # 2. 计算前进线速度 (Translation)
        # ==========================================

        # 这里的算法比较特殊/简化：
        # 它直接使用旋转匹配计算出的“最小差异值 (minDiffIntensityRot)”来估算前进速度。
        # 假设：如果不旋转，但图像差异依然存在（纹理变化剧烈），则认为是前进造成的。
        # 这是一种启发式方法，精度不如光流法。
        transV = minDiffIntensityRot * self.ODO_TRANS_V_SCALE

        # 阈值过滤
        if transV > self.MAX_TRANS_V_THRESHOLD:
            transV = self.PREV_TRANS_V
        else:
            self.PREV_TRANS_V = transV

        # ==========================================
        # 3. 计算垂直/高度速度 (Height/Pitch)
        # ==========================================

        # 裁剪出用于计算高度变化的ROI区域
        subRawImg = rawImg[self.ODO_IMG_HEIGHT_V_Y_RANGE, self.ODO_IMG_HEIGHT_V_X_RANGE]
        subRawImg = np.float32(imresize(subRawImg, self.ODO_IMG_HEIGHT_V_RESIZE_RANGE))

        # --- 旋转补偿 ---
        # 在计算垂直变化前，先根据刚才计算出的水平偏移(minOffsetYawRot)对图像进行对齐。
        # 防止因为水平旋转导致的图像错位影响垂直方向的计算。
        if minOffsetYawRot >= 0:
            subRawImg = subRawImg[:, minOffsetYawRot:]  # 裁掉左边
        else:
            subRawImg = subRawImg[:, : minOffsetYawRot]  # 裁掉右边

        # --- 降维处理 (2D -> 1D) ---
        # 将图像沿X轴（水平方向）求和，得到Y轴方向的强度分布曲线。
        # 任何垂直方向的相机抖动（Pitch）或高度变化都会导致该曲线上下平移。
        imageYSums = np.sum(subRawImg, axis=1)

        # 归一化
        avgIntensity = np.mean(imageYSums)
        imageYSums = imageYSums / avgIntensity

        # 对比当前帧与上一帧的Y轴曲线
        minOffsetHeightV, minDiffIntensityHeight = self.compare_segments(
            imageYSums, self.PREV_HEIGHT_V_IMG_Y_SUMS,
            self.ODO_SHIFT_MATCH_VERT, imageYSums.shape[0]
        )

        # 修正方向符号
        if minOffsetHeightV < 0:
            minDiffIntensityHeight = -minDiffIntensityHeight
        self.OFFSET_HEIGHT_V = minOffsetHeightV

        # 根据不同的运动模式 (model) 处理高度速度
        if model == 0:  # 模式0：平地模式（假设高度变化很小）
            if minOffsetHeightV > 3:  # 只有偏移量很大时才认为有高度变化
                # heightV = self.ODO_HEIGHT_V_SCALE * minDiffIntensityHeight
                heightV = 0  # 代码里强制置0了，说明此模式忽略高度变化
            else:
                heightV = 0
            # heightV = 0

        elif model == 1:  # 模式1：允许向上的速度（如上坡/起飞）
            if minOffsetHeightV > 0:
                heightV = self.ODO_HEIGHT_V_SCALE * minDiffIntensityHeight
            elif self.PREV_HEIGHT_V > 0:  # 保持惯性
                heightV = self.PREV_HEIGHT_V
            else:
                heightV = 0

        else:  # 模式2：允许向下的速度（如下坡/降落）
            if minOffsetHeightV < 0:
                heightV = self.ODO_HEIGHT_V_SCALE * minDiffIntensityHeight
            elif self.PREV_HEIGHT_V < 0:  # 保持惯性
                heightV = self.PREV_HEIGHT_V
            else:
                heightV = 0

        # 阈值过滤
        if abs(heightV) > self.MAX_HEIGHT_V_THRESHOLD:
            heightV = self.PREV_HEIGHT_V
        else:
            self.PREV_HEIGHT_V = heightV

        # 更新上一帧数据
        self.PREV_HEIGHT_V_IMG_Y_SUMS = imageYSums

        # 返回计算出的三个速度分量
        return transV, yawRotV, heightV