import numpy as np
from PIL import Image, ImageDraw

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def _get_interested_regions(img, region_width=12, region_hight=12):
    """
    get 5 interested regions with width of `region_width` and hight of `region_hight` of `img`,
    Considering the datasets we can know that the regions in center of almost all images of dataset
    are valid and informative. Hence, we propose 5 regions from the center of the image and the distribution
    of the regions is like a 'cross', i.e.
    |——————————————————————————————————————————|
    |                                          |
    |                                          |
    |                 |——————|                 |
    |                 |  1   |                 |
    |                 |______|                 |
    |                                          |
    |     |——————|    |——————|    |——————|     |
    |     |  2   |    |  3   |    |  4   |     |
    |     |______|    |______|    |______|     |
    |                                          |
    |                 |——————|                 |
    |                 |  5   |                 |
    |                 |______|                 |
    |                                          |
    |                                          |
    |__________________________________________|
    """
    img = reorder_image(img, 'HWC')
    H, W, _ = img.shape
    width_gap = (W - region_width * 3) // 4
    width_gap = 0 if width_gap < 0 else width_gap
    hight_gap = (H - region_hight * 3) // 4
    hight_gap = 0 if hight_gap < 0 else hight_gap

    r1_start = (hight_gap, width_gap*2+region_width)
    r2_start = (hight_gap*2+region_hight, width_gap)
    r3_start = (hight_gap*2+region_hight, width_gap*2+region_width)
    r4_start = (hight_gap*2+region_hight, width_gap*3+region_width*2)
    r5_start = (hight_gap*3+region_hight*2, width_gap*2+region_width)

    r_starts = [r1_start, r2_start, r3_start, r4_start, r5_start]

    regions = []
    for rr in r_starts:
        regions.append(img[rr[0]: rr[0]+region_hight, rr[1]: rr[1]+region_width])
    return regions


def _draw_interested_regions(img, region_width=12, region_hight=12):
    """
    draw 5 interested regions with width of `region_width` and hight of `region_hight` of `img`,
    Considering the datasets we can know that the regions in center of almost all images of dataset
    are valid and informative. Hence, we propose 5 regions from the center of the image and the distribution
    of the regions is like a 'cross', i.e.
    |——————————————————————————————————————————|
    |                                          |
    |                                          |
    |                 |——————|                 |
    |                 |  1   |                 |
    |                 |______|                 |
    |                                          |
    |     |——————|    |——————|    |——————|     |
    |     |  2   |    |  3   |    |  4   |     |
    |     |______|    |______|    |______|     |
    |                                          |
    |                 |——————|                 |
    |                 |  5   |                 |
    |                 |______|                 |
    |                                          |
    |                                          |
    |__________________________________________|
    """
    H, W = img.shape
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    width_gap = (W - region_width * 3) // 4
    width_gap = 0 if width_gap < 0 else width_gap
    hight_gap = (H - region_hight * 3) // 4
    hight_gap = 0 if hight_gap < 0 else hight_gap

    r1_start = (hight_gap, width_gap*2+region_width)
    r2_start = (hight_gap*2+region_hight, width_gap)
    r3_start = (hight_gap*2+region_hight, width_gap*2+region_width)
    r4_start = (hight_gap*2+region_hight, width_gap*3+region_width*2)
    r5_start = (hight_gap*3+region_hight*2, width_gap*2+region_width)

    r_starts = [r1_start, r2_start, r3_start, r4_start, r5_start]

    for rr in r_starts:
        draw.line((rr[0], rr[1], rr[0]+region_hight, rr[1]), fill=128)
        draw.line((rr[0], rr[1], rr[0], rr[1]+region_width), fill=128)
        draw.line((rr[0]+region_hight, rr[1]+region_width, rr[0]+region_hight, rr[1]), fill=128)
        draw.line((rr[0]+region_hight, rr[1]+region_width, rr[0], rr[1]+region_width), fill=128)

    img.show()
    
    


def _calculate_CR(regions):
    """
    img values: 0-255
    CR: contrast ratio, thus is:
    CR = 20*log10(u_r/u_b),
    where u_r means the mean value of interesting region and u_b denotes the means of background.
    Beacuse of the hardness of difining the background, we set the background as dark, that is u_b = 1
    so, the CR can be formulated as CR = 20*log10(u_r)
    CR指对比度, 但是在对比度计算的时候需要有信号的区域和背景区域，但是难以指定背景区域，而有信号区域可以
    通过上述的`get_interested_regions`方法得到，因此我们简化计算操作，将背景区域视为全黑，也就是说 u_b=1
    """
    cr = 0.
    for r in regions:
        cr += 20 * np.log10(np.mean(r) + 1)
    return cr / len(regions)

def _calculate_CNR(regions):
    """
    img values: 0-255
    CNR: contrast-to-noise
    CNR = |u_r - u_b| / \sqrt{\sigma_r^2 + \sigma_b^2},
    where u_r means the mean value of interesting region and u_b denotes the means of background,
    \sigma_r, \sigma_b indicates the standard deviation of interesting region and background region, respectively.
    Similar to `calculate_CR`, we remove the calculation of background regions, i.e.
    CNR = u_r / \sigma_r
    CNR指对比度噪声比，其被定义为峰值信号强度与背景强度之比。CNR是影像对比度与噪声的比值。是评价影像质量的客观指标。
    对于一幅图像，选取n个感兴趣区域，其中包括背景区域和有效信号区域
    """
    cnr = 0.
    for r in regions:
        cnr += (np.mean(r) / (np.std(r) + 1.))
    return cnr / len(regions)


def calculate_CR_CNR(img):
    regions = _get_interested_regions(img, region_hight=20, region_width=20)
    return _calculate_CR(regions), _calculate_CNR(regions)


# if __name__ == "__main__":
#     import cv2
#     example_path = r'E:\LLIE\Dataset\USenhance\train_datasets\kidney\high_quality\1047.png'
#     example_img = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
#     regions = _draw_interested_regions(example_img, 20, 20)
    
#     cr_low, cnr_low, cr_high, cnr_high = 0., 0., 0., 0.
#     count_low = 0
#     count_high = 0
#     import os
#     root = r'E:\LLIE\Dataset\USenhance\train_datasets'
#     for c in os.listdir(root):
#         c_low_root = os.path.join(root, c, 'low_quality')
#         c_high_root = os.path.join(root, c, 'high_quality')
#         c_low_imgs = [os.path.join(c_low_root, i) for i in os.listdir(c_low_root)]
#         c_high_imgs = [os.path.join(c_high_root, i) for i in os.listdir(c_high_root)]

#         for img in c_low_imgs:
#             img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#             tmp_cr, tmp_cnr = calculate_CR_CNR(img)
#             cr_low += tmp_cr
#             cnr_low += tmp_cnr

#         for img in c_high_imgs:
#             img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#             tmp_cr, tmp_cnr = calculate_CR_CNR(img)
#             cr_high += tmp_cr
#             cnr_high += tmp_cnr

#         count_low += len(c_low_imgs)
#         count_high += len(c_high_imgs)

#     cr_low = cr_low / count_low
#     cnr_low = cnr_low / count_low
#     cr_high = cr_high / count_high
#     cnr_high = cnr_high / count_high

#     print("Low quality: CR: %.3f, CNR: %.3f" % (cr_low, cnr_low))
#     print("High quality: CR: %.3f, CNR: %.3f" % (cr_high, cnr_high))


