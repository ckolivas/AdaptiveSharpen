#!/usr/bin/python3
#
# Copyright 2025 Con Kolivas kernel@kolivas.org
#
# An adaptive sharpening algorithm for finishing planetary images already
# wavelet sharpened
#
# Uses deconvolution of varied strength dependent on the local contrast as
# sharpened stacked images can tolerate less sharpening in low contrast areas
# before sharpening noise is generated
#
# Automatic detection of the optimal max_strength is a blunt tool that expects
# an initially stacked image with a max histogram stretched to 70%

import argparse
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import generic_filter
import cv2
from skimage.color import rgb2lab, lab2rgb
from numpy.lib.stride_tricks import sliding_window_view

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

def rgb2oklab(rgb):
    """Convert gamma-corrected sRGB [0,1] to Oklab (assumes D65 whitepoint)."""
    # Linearize
    linear = srgb_to_linear(rgb)

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
    return oklab * 100

def oklab2rgb(oklab):
    oklab /= 100
    """Convert Oklab to gamma-corrected sRGB [0,1]."""
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

    # Gamma correct
    srgb = linear_to_srgb(linear)
    return np.clip(srgb, 0, 1)  # Ensure [0,1]

def rgb2lum(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

def std_windowed(img, win_size):
    win_h, win_w = win_size
    win_view = sliding_window_view(img, (win_h, win_w), axis=(0, 1))
    return win_view.std(axis=(-2, -1))

def main():
    parser = argparse.ArgumentParser(description='Apply adaptive Lucy-Richardson deconvolution on luminance channel of a 16-bit PNG image.')
    parser.add_argument('input', help='Input PNG file')
    parser.add_argument('output', help='Output 16-bit PNG file')
    parser.add_argument('--max_strength', type=float, default=None, help='Maximum deconvolution strength (default: auto; higher values for more aggressive sharpening)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output (saves contrast map)')
    parser.add_argument('--no_contrast', action='store_true', help='Apply fixed deconvolution strength without contrast adaptation')
    parser.add_argument('--oklab', action='store_true', help='Use OKlab instead of cielab deconvolution')
    parser.add_argument('--rgb', action='store_true', help='Use RGB instead of cielab deconvolution')
    parser.add_argument('--denoise', action='store_true', help='Denoise dark pixels for when artefacts in black appear')
    args = parser.parse_args()

    if args.oklab & args.rgb:
        raise ValueError("Cannot use both oklab and rgb")

    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    # Convert BGR(A) to RGB
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 3 and image.shape[-1] == 2:
        image = image[..., 0]

    if image.dtype == np.uint16:
        max_intensity = 65535.0
    elif image.dtype == np.uint8:
        max_intensity = 255.0
    else:
        raise ValueError("Unsupported image dtype")

    rgb = image.astype(np.float32) / max_intensity

    is_colour = len(rgb.shape) == 3
    if not is_colour:
        rgb = np.dstack((rgb, rgb, rgb))
    print("Processing ", args.input)

    psf = generate_moffat_kernel(gamma=1.0, beta=2.0, size=21)

    if not is_colour:
        lab = rgb
        rgb = lab[:, :, np.newaxis]
    else:
        # Use LAB for luminance deconvolution
        if args.oklab:
            lab = rgb2oklab(rgb)
        else:
            lab = rgb2lab(rgb)

    if args.rgb:
        lum = rgb2lum(rgb)
    else:
        lum = lab[..., 0] / 100.0

    original_lum = lum.copy()
    if args.denoise:
        bg = np.percentile(original_lum, 5)
    else:
        bg = np.median(original_lum)
    lum -= bg
    lum = np.maximum(lum, 0)

    window_size = 7
    contrast =std_windowed(lum, (window_size, window_size))
    contrast= np.pad(contrast,window_size//2)

    contrast_min = contrast.min()
    contrast_max = contrast.max()
    contrast_norm = (contrast - contrast_min) / (contrast_max - contrast_min + 1e-10)

    # Generate debug contrast map if --debug is set
    if args.debug:
        debug_img = np.clip(contrast_norm * 65535, 0, 65535).astype(np.uint16)
        cv2.imwrite(args.output + '_debug.png', debug_img)

    def compute_sharpened(strength, fixed=False):
        max_val = 2 * lum.max()
        current = lum.copy()

        conv = fftconvolve(current, psf, mode='same')
        relative = lum / np.maximum(conv, 1e-12)
        correction = fftconvolve(relative, psf, mode='same')
        if fixed:
            local_strength = strength
        else:
            local_strength = strength * (contrast_norm ** 0.5)
        damped_correction = 1 + local_strength * (correction - 1)
        current = current * damped_correction
        current = np.clip(current, 0, max_val)

        lum_sharp = np.maximum(current, 0)
        lum_sharp += bg

        lab_sharp = lab.copy()
        ratio = lum_sharp / np.maximum(original_lum, 1e-12)
        if args.denoise:
            ratio = np.clip(ratio, 0.5, 2.0)
        if not args.rgb:
            lab_sharp[..., 0] = lum_sharp * 100.0
            lab_sharp[..., 0] *= ratio
        if args.oklab:
            rgb_sharp = oklab2rgb(lab_sharp)
        elif args.rgb:
            rgb_sharp = rgb * ratio[:, :, np.newaxis]
        else:
            rgb_sharp = lab2rgb(lab_sharp)
        return np.clip(rgb_sharp * 65535, 0, 65535).astype(np.uint16)

    if args.no_contrast:
        strength = args.max_strength if args.max_strength is not None else 10.0
        out_img = compute_sharpened(strength, fixed=True)
        print(f"Used max_strength: {strength}")
    else:
        if args.max_strength is None:
            strength = 1.0
            best_strength = 1.0
            best_out_img = compute_sharpened(strength)
            while True:
                out_img = compute_sharpened(strength)
                # Compute max brightness as percentage
                if is_colour:
                    rgb_out = out_img.astype(np.float32) / 65535.0
                    lab_out = rgb2lab(rgb_out)
                    max_brightness = np.max(lab_out[..., 0])
                else:
                    max_brightness = np.max(out_img.astype(np.float32) / 65535.0) * 100
                if max_brightness > 95:
                    out_img = best_out_img
                    print(f"Used max_strength: {best_strength}")
                    break
                best_strength = strength
                best_out_img = out_img
                strength += 1.0  # Increase by 1 each time
                if strength > 50:  # Safety cap to prevent infinite loop
                    print(f"Used max_strength: {best_strength}")
                    break
        else:
            out_img = compute_sharpened(args.max_strength)
            print(f"Used max_strength: {args.max_strength}")

    if not is_colour:
        out_img = out_img[:, :, 0]
        cv2.imwrite(args.output, out_img)  # Grayscale
    else:
        cv2.imwrite(args.output, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()
