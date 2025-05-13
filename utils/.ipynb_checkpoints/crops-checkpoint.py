import numpy as np
from PIL import Image
import random
from scipy import stats
from skimage import feature, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch
from torchvision.transforms import CenterCrop

############################################################### TEXTURE CROP ###############################################################

def texture_crop(image, stride=224, window_size=224, metric='he', position='top', n=10, drop = False):
    cropped_images = []
    images = []

    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))
    
    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray)
        if metric == 'sd':
            m = np.std(crop_gray / 255.0)
        elif metric == 'ghe':
            m = histogram_entropy_response(crop_gray / 255.0)
        elif metric == 'le':
            m = local_entropy_response(crop_gray)
        elif metric == 'ac':
            m = autocorrelation_response(crop_gray / 255.0)
        elif metric == 'td':
            m = texture_diversity_response(crop_gray / 255.0)
        images.append((crop, m))

    images.sort(key=lambda x: x[1], reverse=True)
    
    if position == 'top':
        texture_images = [img for img, _ in images[:n]]
    elif position == 'bottom':
        texture_images = [img for img, _ in images[-n:]]

    repeat_images = texture_images.copy()
    while len(texture_images) < n:
        texture_images.append(repeat_images[len(texture_images) % len(repeat_images)])

    return texture_images


def autocorrelation_response(image_array):
    """
    Calculates the average autocorrelation of the input image.
    """
    f = np.fft.fft2(image_array, norm='ortho')
    power_spectrum = np.abs(f) ** 2
    acf = np.fft.ifft2(power_spectrum, norm='ortho').real
    acf = np.fft.fftshift(acf)
    acf /= acf.max()
    acf = np.mean(acf)

    return acf

def histogram_entropy_response(image):
    """
    Calculates the entropy of the image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True) 
    prob_dist = histogram / histogram.sum()
    entr = stats.entropy(prob_dist + 1e-7, base=2)    # Adding a small value (1e-7) to avoid log(0)

    return entr

def local_entropy_response(image):
    """
    Calculates the spatial entropy of the image using a local entropy filter.
    """
    entropy_image = entropy(image, disk(10))  
    mean_entropy = np.mean(entropy_image)

    return mean_entropy

def texture_diversity_response(image):
    M = image.shape[0]  
    l_div = 0

    for i in range(M):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i, j + 1])

    # Vertical differences
    for i in range(M - 1):
        for j in range(M):
            l_div += abs(image[i, j] - image[i + 1, j])

    # Diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i + 1, j + 1])

    # Counter-diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i + 1, j] - image[i, j + 1])

    return l_div


############################################################## THRESHOLDTEXTURECROP ##############################################################

def threshold_texture_crop(image, stride=224, window_size=224, threshold=5, drop = False):
    cropped_images = []
    texture_images = []
    images = []

    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))

    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray) / 255.0
        
        histogram, _ = np.histogram(crop_gray.flatten(), bins=256, range=(0, 1), density=True) 
        prob_dist = histogram / histogram.sum()
        m = stats.entropy(prob_dist + 1e-7, base=2)
        if m > threshold: 
            texture_images.append(crop)

    if len(texture_images) == 0:
        texture_images = [CenterCrop(image)]

    return texture_images

def combine_texture_crops(texture_images, grid_size = 4):
    """
    将纹理裁剪图像列表组合成一个 n x n 的正方形图像。

    Args:
        texture_images: 包含 PIL.Image 对象的列表。
        n: 正方形网格的块数（每行和每列的块数）。

    Returns:
        一个 PIL.Image 对象，表示组合后的正方形图像。
    """
    if not texture_images:
        return None

    first_image = texture_images[0]
    image_width, image_height = first_image.size
    combined_width = grid_size * image_width
    combined_height = grid_size * image_height
    combined_image = Image.new('RGB', (combined_width, combined_height))

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            if index < len(texture_images):
                x_offset = j * image_width
                y_offset = i * image_height
                combined_image.paste(texture_images[index], (x_offset, y_offset))
            else:
                # 如果提供的 texture_images 不足 grid_size*grid_size 张，则使用最后一张填充
                last_image = texture_images[-1]
                x_offset = j * image_width
                y_offset = i * image_height
                combined_image.paste(last_image, (x_offset, y_offset))

    return combined_image

# 新增：将 texture_crop 封装为可直接用于 Compose 的 Transform
class TextureCrop(object):
    """
    可作为 torchvision transform 的纹理裁剪类，返回组合后的 PIL.Image。
    """
    def __init__(self, stride, window_size, metric='he', position='top', n=16, drop=False):
        self.stride = stride
        self.window_size = window_size
        self.metric = metric
        self.position = position
        self.n = n
        self.drop = drop
        # fallback 中心裁剪，用于图像小于 window_size 时自动 pad_if_needed
        self._fallback = CenterCrop(window_size)

    def __call__(self, image):
        # Calculate grid_dim based on self.n, used for consistent output size
        # Since self.n >= 1 (default is 16 in __init__), grid_dim will be >= 1.
        grid_dim = int(np.ceil(np.sqrt(self.n)))

        if image.width < self.window_size or image.height < self.window_size:
            single_center_crop = self._fallback(image) # This is window_size x window_size
            
            # Create a list of crops for the fallback, tiling the single_center_crop
            # to fill the grid_dim * grid_dim grid.
            num_fallback_crops = grid_dim * grid_dim
            fallback_crops = [single_center_crop] * num_fallback_crops
            
            # Combine these fallback crops into an image of size 
            # (grid_dim * window_size, grid_dim * window_size)
            return combine_texture_crops(fallback_crops, grid_size=grid_dim)

        # Proceed with normal texture cropping
        crops = texture_crop(
            image,
            stride=self.stride,
            window_size=self.window_size,
            metric=self.metric,
            position=self.position,
            n=self.n, 
            drop=self.drop
        )
        
        # 'crops' contains self.n images. 
        # combine_texture_crops will tile if len(crops) < grid_dim * grid_dim.
        combined_image = combine_texture_crops(crops, grid_size=grid_dim)
        return combined_image