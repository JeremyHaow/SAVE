import numpy as np  # 导入NumPy库，用于高效的数组运算
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
import random  # 导入random模块，用于生成随机数
from scipy import stats  # 导入scipy的stats模块，用于统计学函数
from skimage import feature, filters  # 导入skimage的特征提取和过滤器模块
from skimage.filters.rank import entropy  # 导入skimage中的熵计算函数
from skimage.morphology import disk  # 导入skimage中的磁盘结构元素，用于形态学操作
import torch  # 导入PyTorch库
from torchvision.transforms import CenterCrop  # 导入PyTorch的中心裁剪变换

############################################################### TEXTURE CROP ###############################################################

def texture_crop(image, stride=224, window_size=224, metric='he', position='top', n=10, drop = False):
    """
    基于纹理特征对图像进行裁剪
    
    参数:
    image: PIL图像对象，输入的原始图像
    stride: 滑动窗口的步长，默认224像素
    window_size: 裁剪窗口的大小，默认224x224像素
    metric: 纹理评估指标，可选'sd'(标准差),'ghe'(直方图熵),'le'(局部熵),'ac'(自相关),'td'(纹理多样性)
    position: 选择纹理得分排序的位置，'top'选择得分最高的，'bottom'选择得分最低的
    n: 返回的裁剪图像数量
    drop: 是否丢弃边缘不完整的裁剪块，默认False
    
    返回:
    texture_images: 列表，包含n个基于纹理特征选择的裁剪图像
    """
    cropped_images = []  # 存储所有裁剪后的图像
    images = []  # 存储裁剪图像及其纹理得分的元组
    
    # 使用滑动窗口方法裁剪图像，步长为stride
    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))
    
    # 如果不丢弃边缘区域，处理图像边缘的剩余部分
    if not drop:
        x = x + stride  # 更新x坐标
        y = y + stride  # 更新y坐标
        
        # 处理右边缘
        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        
        # 处理底边缘
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        
        # 处理右下角
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))
    
    # 计算每个裁剪图像的纹理特征得分
    for crop in cropped_images:
        crop_gray = crop.convert('L')  # 转换为灰度图
        crop_gray = np.array(crop_gray)  # 转换为NumPy数组
        
        # 根据指定的度量方法计算纹理得分
        if metric == 'sd':  # 标准差
            m = np.std(crop_gray / 255.0)
        elif metric == 'ghe':  # 全局直方图熵
            m = histogram_entropy_response(crop_gray / 255.0)
        elif metric == 'le':  # 局部熵
            m = local_entropy_response(crop_gray)
        elif metric == 'ac':  # 自相关
            m = autocorrelation_response(crop_gray / 255.0)
        elif metric == 'td':  # 纹理多样性
            m = texture_diversity_response(crop_gray / 255.0)
        
        images.append((crop, m))  # 将裁剪图像和其纹理得分添加到列表中
    
    # 根据纹理得分对图像进行排序（降序）
    images.sort(key=lambda x: x[1], reverse=True)
    
    # 根据position参数选择纹理得分排名的图像
    if position == 'top':
        texture_images = [img for img, _ in images[:n]]  # 选择得分最高的n个图像
    elif position == 'bottom':
        texture_images = [img for img, _ in images[-n:]]  # 选择得分最低的n个图像
    
    # 如果获取的图像少于要求的n个，则重复已有图像直到达到n个
    repeat_images = texture_images.copy()
    while len(texture_images) < n:
        texture_images.append(repeat_images[len(texture_images) % len(repeat_images)])
    
    return texture_images  # 返回选择的n个裁剪图像


def autocorrelation_response(image_array):
    """
    计算输入图像的平均自相关
    
    自相关可以反映图像中重复模式的存在，是一种纹理分析方法
    
    参数:
    image_array: 输入图像的NumPy数组
    
    返回:
    acf: 图像的平均自相关系数
    """
    f = np.fft.fft2(image_array, norm='ortho')  # 对图像进行二维傅里叶变换
    power_spectrum = np.abs(f) ** 2  # 计算功率谱（幅度平方）
    acf = np.fft.ifft2(power_spectrum, norm='ortho').real  # 通过功率谱的逆傅里叶变换获取自相关函数（取实部）
    acf = np.fft.fftshift(acf)  # 将零频率分量移至中心
    acf /= acf.max()  # 归一化自相关函数
    acf = np.mean(acf)  # 计算平均自相关系数
    
    return acf


def histogram_entropy_response(image):
    """
    计算图像的熵值
    
    熵是图像复杂度或信息量的度量，高熵值表示图像中包含丰富的信息
    
    参数:
    image: 输入图像的NumPy数组
    
    返回:
    entr: 图像的熵值
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True)  # 计算图像灰度直方图
    prob_dist = histogram / histogram.sum()  # 计算概率分布
    entr = stats.entropy(prob_dist + 1e-7, base=2)  # 计算熵值，添加1e-7以避免log(0)
    
    return entr


def local_entropy_response(image):
    """
    使用局部熵滤波器计算图像的空间熵
    
    局部熵考虑了像素的局部邻域，能更好地表示纹理的空间分布
    
    参数:
    image: 输入图像的NumPy数组
    
    返回:
    mean_entropy: 局部熵图像的平均值
    """
    entropy_image = entropy(image, disk(10))  # 使用半径为10的磁盘结构元素计算局部熵
    mean_entropy = np.mean(entropy_image)  # 计算局部熵图像的平均值
    
    return mean_entropy


def texture_diversity_response(image):
    """
    计算图像的纹理多样性
    
    通过计算相邻像素之间的差异来量化图像的纹理复杂度
    包括水平、垂直和对角线方向的差异
    
    参数:
    image: 输入图像的NumPy数组
    
    返回:
    l_div: 纹理多样性得分
    """
    M = image.shape[0]  # 获取图像的高度（假设是正方形）
    l_div = 0  # 初始化纹理多样性得分
    
    # 计算水平方向的像素差异
    for i in range(M):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i, j + 1])
    
    # 计算垂直方向的像素差异
    for i in range(M - 1):
        for j in range(M):
            l_div += abs(image[i, j] - image[i + 1, j])
    
    # 计算对角线方向（左上到右下）的像素差异
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i + 1, j + 1])
    
    # 计算反对角线方向（右上到左下）的像素差异
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i + 1, j] - image[i, j + 1])
    
    return l_div  # 返回纹理多样性得分


############################################################## THRESHOLDTEXTURECROP ##############################################################

def threshold_texture_crop(image, stride=224, window_size=224, threshold=5, drop = False):
    """
    基于熵阈值的纹理裁剪方法
    
    仅选择熵值高于指定阈值的区域，这些区域通常包含更丰富的纹理信息
    
    参数:
    image: PIL图像对象，输入的原始图像
    stride: 滑动窗口的步长，默认224像素
    window_size: 裁剪窗口的大小，默认224x224像素
    threshold: 熵值阈值，只保留熵值大于此阈值的裁剪图像
    drop: 是否丢弃边缘不完整的裁剪块，默认False
    
    返回:
    texture_images: 列表，包含熵值高于阈值的裁剪图像
    """
    cropped_images = []  # 存储所有裁剪后的图像
    texture_images = []  # 存储熵值高于阈值的裁剪图像
    images = []  # 存储裁剪图像及其熵值的元组
    
    # 使用滑动窗口方法裁剪图像
    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))
    
    # 如果不丢弃边缘区域，处理图像边缘的剩余部分
    if not drop:
        x = x + stride  # 更新x坐标
        y = y + stride  # 更新y坐标
        
        # 处理右边缘
        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        
        # 处理底边缘
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        
        # 处理右下角
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))
    
    # 计算每个裁剪图像的熵值并筛选
    for crop in cropped_images:
        crop_gray = crop.convert('L')  # 转换为灰度图
        crop_gray = np.array(crop_gray) / 255.0  # 转换为NumPy数组并归一化
        
        # 计算图像的直方图熵
        histogram, _ = np.histogram(crop_gray.flatten(), bins=256, range=(0, 1), density=True) 
        prob_dist = histogram / histogram.sum()  # 计算概率分布
        m = stats.entropy(prob_dist + 1e-7, base=2)  # 计算熵值
        
        # 只保留熵值高于阈值的图像
        if m > threshold: 
            texture_images.append(crop)
    
    # 如果没有图像的熵值高于阈值，则返回原图像的中心裁剪
    if len(texture_images) == 0:
        texture_images = [CenterCrop(image)]
    
    return texture_images  # 返回选择的裁剪图像

# 新增：将 texture_crop 封装为可直接用于 Compose 的 Transform
class TextureCrop(object):
    """
    可作为 torchvision transform 的纹理裁剪类，返回单张 PIL.Image。
    """
    def __init__(self, stride, window_size, metric='he', position='top', n=1, drop=False):
        self.stride = stride
        self.window_size = window_size
        self.metric = metric
        self.position = position
        self.n = n
        self.drop = drop
        # fallback 中心裁剪，用于图像小于 window_size 时自动 pad_if_needed
        self._fallback = CenterCrop(window_size)

    def __call__(self, image):
        # 如果图像尺寸不足，直接做中心裁剪
        if image.width < self.window_size or image.height < self.window_size:
            return self._fallback(image)
        # 否则调用原函数并取第一张
        crops = texture_crop(
            image,
            stride=self.stride,
            window_size=self.window_size,
            metric=self.metric,
            position=self.position,
            n=self.n,
            drop=self.drop
        )
        return crops[0]