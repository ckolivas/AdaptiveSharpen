#!/usr/bin/python3
#Shared code version of AdaptiveSharpen for use with WS3

import argparse
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import generic_filter
import imageio.v2 as imageio
import cv2
from numpy.lib.stride_tricks import sliding_window_view

from matplotlib import pyplot as plt

def generate_moffat_kernel(gamma=1.0, beta=2.0, size=21):
    half = size // 2
    x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
    r = np.sqrt(x**2 + y**2)
    psf = (1 + (r / gamma)**2) ** (-beta)
    psf /= psf.sum()
    return psf

def srgb_to_linear(rgb):
    """Linearize gamma-corrected sRGB [0,1] to linear RGB."""
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )

def linear_to_srgb(linear):
    """Gamma-correct linear RGB to sRGB [0,1]."""
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1 / 2.4)) - 0.055
    )

def linearrgb_to_oklab(linear):
    # Linear RGB to LMS (first matrix)
    M1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])
    lms = np.einsum('ij,...j->...i', M1, linear)

    # Cube root
    lms_prime = np.cbrt(lms)

    # LMS' to Oklab (second matrix)
    M2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])
    oklab = np.einsum('ij,...j->...i', M2, lms_prime)
    return oklab

def oklab_to_linearrgb(oklab):
    # Oklab to LMS' (inverse second matrix)
    M2_inv = np.array([
        [1.0000000000,  0.3963377774,  0.2158037573],
        [1.0000000000, -0.1055613458, -0.0638541728],
        [1.0000000000, -0.0894841775, -1.2914855480]
    ])
    lms_prime = np.einsum('ij,...j->...i', M2_inv, oklab)

    # Cube
    lms = lms_prime ** 3

    # LMS to linear RGB (inverse first matrix)
    M1_inv = np.array([
        [ 4.0767416621, -3.3077115913,  0.2309699292],
        [-1.2684380046,  2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147,  1.7076147010]
    ])
    linear = np.einsum('ij,...j->...i', M1_inv, lms)
    linear = np.maximum(linear, 0)  # Clip negatives to prevent invalid power

    return np.clip(linear, 0, 1)  # Ensure [0,1]

def linear_rgb2lum(rgb):
    return 0.212671 * rgb[:, :, 0] + 0.715160 * rgb[:, :, 1] + 0.072169 * rgb[:, :, 2]

def std_windowed(img, win_size):
    win_h, win_w = win_size
    win_view = sliding_window_view(img, (win_h, win_w), axis=(0, 1))
    return win_view.std(axis=(-2, -1))

def print_stats(img):
    print('min:', img.min())
    print('max:', img.max())
    print('mean:', np.mean(img))
    print('p10:', np.percentile(img,10))
    print('p50:', np.percentile(img,50))
    print('p90:', np.percentile(img,90))

#function with several defaults 
def deconvolve(image, max_intensity, max_strength=1, no_contrast=False):
   
   # fig, ax = plt.subplots(3, 2,figsize=(10,20))
    
    #Apply a fudge factor to approximate linear luminance in ok linear
    #luminance and give more gradation
    strength = max_strength / 3.14

    #when not specified use filetype to deduct default max_intensity
    if max_intensity is None:
        if image.dtype == np.uint16:
            max_intensity = 65535.0
        elif image.dtype == np.uint8:
            max_intensity = 255.0
        elif image.dtype == np.float32: 
            max_intensity = 1.0
        else:
            raise ValueError("Unsupported image dtype")

    print('image')
    print_stats(image)

    # Handle possible alpha channel
    if len(image.shape) == 3 and image.shape[-1] == 4:
        alpha = image[..., 3].astype(np.float32) / max_intensity
        image = image[..., :3]
    else:
        alpha = None
    
    srgb = image.astype(np.float32) / max_intensity
    rgb = srgb_to_linear(srgb)
        
   # ax[0,0].imshow(rgb)

    psf = generate_moffat_kernel(gamma=1.0, beta=2.0, size=21)
    print('psf')
    print_stats(psf)
    psf_mirror = np.flip(psf)

    #START OF LUMINANCE ONLY PROCESSING
    #lum into 0..1 range after rgb2lab transformation
    lum = linear_rgb2lum(rgb)
    
    print('lum')
    print_stats(lum)
    #WARNING: does not detect alpha images !!
    is_color = len(rgb.shape) == 3 and rgb.shape[-1] == 3

    original_lum = lum.copy()
    lum = np.maximum(lum, 0)
    
    print('np max lum')
    print_stats(lum)

    window_size = 7
    contrast = std_windowed(lum, (window_size, window_size))
    contrast = np.pad(contrast,window_size//2)

    contrast_min = contrast.min()
    contrast_max = contrast.max()
    contrast_norm = (contrast - contrast_min) / (contrast_max - contrast_min + 1e-10)

   # ax[1,0].imshow(contrast_norm)

    print('contrast_norm')
    print_stats(contrast_norm)
    
    # Generate debug contrast map if --debug is set
    #if debug:
    #    debug_img = np.clip(contrast_norm * 65535, 0, 65535).astype(np.uint16)
    #    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR).astype(np.uint16)
    #    cv2.imwrite(args.output + '_debug.png', debug_img)
    
    #CORB: removed the option that max_strength was not set as it defaults to 1 in the functioncall
    # and also simplified further as fixed equals no_contrast
    max_val = 2 * lum.max()
    max_val = np.clip(max_val, 0, 1)
    current = lum.copy()

    # Single iteration with local strength
    conv = fftconvolve(current, psf, mode='same')
    relative = lum / np.maximum(conv, 1e-12)
    correction = fftconvolve(relative, psf_mirror, mode='same')
    
    if no_contrast:
        local_strength = strength
    else:
        local_strength = strength * (contrast_norm ** 0.5)
        
    damped_correction = 1 + local_strength * (correction - 1)
    current = current * damped_correction

    lum_sharp = np.maximum(current, 0)
    #ax[2,0].imshow(current)

    ratio = lum_sharp / np.maximum(original_lum, 1e-12)
    oklab_linear = linearrgb_to_oklab(rgb)
    oklab_linear[..., 0] *= ratio
    print('oklab_linear')
    print_stats(oklab_linear)

    #Convert back to linear rgb
    rgb_sharp = oklab_to_linearrgb(oklab_linear)
    #ax[2,1].imshow(current)
    
    rgb_sharp = np.clip(rgb_sharp, 0, max_val)
    print('rgb_sharp')
    print_stats(rgb_sharp)

    #convert to srgb
    out_srgb = linear_to_srgb(rgb_sharp)
    out_img = out_srgb.astype(np.float32)

    print('out_img')
    print_stats(out_img)
       
    if not is_color:
        out_img = out_img[:, :, 0]

    # Convert all transparent pixels to black if alpha exists
    if alpha is not None:
        out_img[alpha == 0] = 0
        
    # ax[0,1].imshow(out_img)
   
    #convert image back into original range
    return out_img*max_intensity
    
     


def main():
    parser = argparse.ArgumentParser(description='Apply adaptive Lucy-Richardson deconvolution on luminance channel of a 16-bit PNG image.')
    parser.add_argument('input', help='Input PNG file')
    parser.add_argument('output', help='Output 16-bit PNG file')
    parser.add_argument('--max_strength', type=float, default=10, help='Maximum deconvolution strength (default: auto; higher values for more aggressive sharpening)')
    #parser.add_argument('--debug', action='store_true', help='Enable debug output (saves contrast map)')
    parser.add_argument('--no_contrast', action='store_true', help='Apply fixed deconvolution strength without contrast adaptation')
    args = parser.parse_args()

    #no_contrast=False
    #fname= '2023-09-23-1621_9-CK-L3-Jup_expand000001_P7_ap65g.png'
    
    image=cv2.imread(args.input,-1)
    #create a normal channel order from cv2 loading
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    #out_img=deconvolve(image, max_intensity=65535, max_strength=15)
    out_img=deconvolve(image, None, args.max_strength, args.no_contrast)

    #recreate normal channel order for further processing
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR).astype(np.uint16)
    cv2.imwrite(args.output,out_img)
    

main()
