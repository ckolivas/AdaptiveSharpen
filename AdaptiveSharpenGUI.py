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

import argparse
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import generic_filter
import cv2
from numpy.lib.stride_tricks import sliding_window_view
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sys

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
    """Convert gamma-corrected sRGB [0,1] to Oklab (assumes D65 whitepoint)."""

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

    return np.clip(linear, 0, 1)  # Ensure [0,1]

def linear_rgb2lum(rgb):
    return 0.212671 * rgb[:, :, 0] + 0.715160 * rgb[:, :, 1] + 0.072169 * rgb[:, :, 2]

def std_windowed(img, win_size):
    win_h, win_w = win_size
    win_view = sliding_window_view(img, (win_h, win_w), axis=(0, 1))
    return win_view.std(axis=(-2, -1))

def process_image(input_path, output_path, max_strength=None, debug=False, no_contrast=False, rgb_sharpen = False, noisy=False):

    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
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

    clipped = False

    srgb = image.astype(np.float32) / max_intensity
    max_lum = np.max(srgb)
    if noisy:
        lum_boost = 1.125
    else:
        lum_boost = 1.25
    lum_cap = 1.0 / lum_boost

    if max_strength == None and max_lum >= lum_cap:
        print("Decreasing luminance on bright image with max luminance of ", max_lum)
        srgb *= 0.75 / max_lum
        max_lum = 0.75
    rgb = srgb_to_linear(srgb)  # Convert to linear pixels after input

    is_colour = len(rgb.shape) == 3
    if not is_colour:
        rgb = np.dstack((rgb, rgb, rgb))
    print("Processing ", input_path)

    psf = generate_moffat_kernel(gamma=1.0, beta=2.0, size=21)

    lum = linear_rgb2lum(rgb)

    original_lum = lum.copy()
    lum = np.maximum(lum, 0)

    if rgb_sharpen:
        img = np.maximum(rgb, 0)

    window_size = 7
    contrast = std_windowed(lum, (window_size, window_size))
    contrast = np.pad(contrast, window_size//2)

    contrast_min = contrast.min()
    contrast_max = contrast.max()
    contrast_norm = (contrast - contrast_min) / (contrast_max - contrast_min + 1e-10)

    # Generate debug contrast map if debug is set
    if debug:
        debug_img = np.clip(contrast_norm * 65535, 0, 65535).astype(np.uint16)
        cv2.imwrite(output_path + '_debug.png', debug_img)

    def compute_sharpened(strength, fixed=False):
        nonlocal clipped

        clipped = False
        if rgb_sharpen:
            rgb_sharp = np.zeros_like(img)
            for i in range(3):
                current = img[..., i].copy()

                conv = fftconvolve(current, psf, mode='same')
                relative = current / np.maximum(conv, 1e-12)
                correction = fftconvolve(relative, psf, mode='same')
                if fixed:
                    local_strength = strength
                else:
                    local_strength = strength * (contrast_norm ** 0.5)
                damped_correction = 1 + local_strength * (correction - 1)
                current = current * damped_correction

                channel_sharp = np.maximum(current, 0)
                ratio = channel_sharp / np.maximum(rgb[..., i], 1e-12)
                rgb_sharp[..., i] = rgb[..., i] * ratio
        else:
            #Apply a fudge factor to approximate linear luminance in ok linear
            #luminance and give more gradation
            strength = strength / 3.14

            oklab_linear = linearrgb_to_oklab(rgb)
            current = lum.copy()

            conv = fftconvolve(current, psf, mode='same')
            relative = current / np.maximum(conv, 1e-12)
            correction = fftconvolve(relative, psf, mode='same')
            if fixed:
                local_strength = strength
            else:
                local_strength = strength * (contrast_norm ** 0.5)
            damped_correction = 1 + local_strength * (correction - 1)
            current = current * damped_correction

            lum_sharp = np.maximum(current, 0)
            ratio = lum_sharp / np.maximum(original_lum, 1e-12)
            oklab_linear[..., 0] *= ratio
            rgb_sharp = oklab_to_linearrgb(oklab_linear)

        local_max = np.max(rgb_sharp)
        if local_max > 1:
            clipped = True
            rgb_sharp *= 1 / local_max

        return rgb_sharp

    if max_strength is None:
        strength = 1.0
        best_strength = 1.0
        best_out_img = compute_sharpened(strength, no_contrast)
        while True:
            out_linear = compute_sharpened(strength, no_contrast)
            out_srgb = linear_to_srgb(out_linear)
            out_maxlum = np.max(out_srgb)
            if out_maxlum > max_lum * lum_boost:
                print(f"Used max_strength: {best_strength}")
                break
            best_strength = strength
            best_out_img = out_linear
            strength += 1.0  # Increase by 1 each time
            if strength > 200:  # Safety cap to prevent infinite loop
                print(f"Used max_strength: {best_strength}")
                break
        out_linear = best_out_img
    else:
        out_linear = compute_sharpened(max_strength, no_contrast)
        print(f"Used max_strength: {max_strength}")

    out_srgb = linear_to_srgb(out_linear) * 65535
    out_img = out_srgb.astype(np.uint16)

    if not is_colour:
        out_img = out_img[:, :, 0]
        cv2.imwrite(output_path, out_img)  # Grayscale
    else:
        cv2.imwrite(output_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser(description='Apply adaptive Lucy-Richardson deconvolution on luminance channel of a 16-bit PNG image.')
    parser.add_argument('input', help='Input PNG file')
    parser.add_argument('output', help='Output 16-bit PNG file')
    parser.add_argument('--max_strength', type=float, default=None, help='Maximum deconvolution strength (default: auto; higher values for more aggressive sharpening)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output (saves contrast map)')
    parser.add_argument('--no_contrast', action='store_true', help='Apply fixed deconvolution strength without contrast adaptation')
    parser.add_argument('--rgb', action='store_true', help='Use RGB instead of cielab deconvolution')
    parser.add_argument('--noisy', action='store_true', help='Use less max_strength in auto for noisy images')
    args = parser.parse_args()

    process_image(args.input, args.output, max_strength=args.max_strength, debug=args.debug, no_contrast=args.no_contrast, rgb_sharpen=args.rgb, noisy=args.noisy)

def run_gui():
    def select_input():
        input_path.set(filedialog.askopenfilename(filetypes=[("PNG files", "*.png")]))

    def select_output():
        output_path.set(filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")]))

    def run_processing():
        try:
            # Collect values
            ms = float(max_strength_entry.get()) if max_strength_entry.get() else None
            process_image(input_path.get(), output_path.get(), max_strength=ms,
                          debug=debug_var.get(), no_contrast=no_contrast_var.get(),
                          rgb_sharpen=rgb_var.get(), noisy=noisy_var.get())
            messagebox.showinfo("Success", "Processing complete!")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def start_processing():
        if not input_path.get() or not output_path.get():
            messagebox.showwarning("Input Required", "Select input and output files.")
            return
        threading.Thread(target=run_processing).start()  # Run in background

    root = tk.Tk()
    root.title("Adaptive Sharpen GUI")
    root.geometry("400x500")  # Simple size

    # Input file
    tk.Label(root, text="Input PNG:").pack()
    input_path = tk.StringVar()
    tk.Entry(root, textvariable=input_path, width=50).pack()
    tk.Button(root, text="Browse", command=select_input).pack()

    # Output file
    tk.Label(root, text="Output PNG:").pack()
    output_path = tk.StringVar()
    tk.Entry(root, textvariable=output_path, width=50).pack()
    tk.Button(root, text="Browse", command=select_output).pack()

    # Options
    tk.Label(root, text="Max Strength (optional):").pack()
    max_strength_entry = tk.Entry(root)
    max_strength_entry.pack()

    debug_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Debug (save contrast map)", variable=debug_var).pack()

    no_contrast_var = tk.BooleanVar()
    tk.Checkbutton(root, text="No Contrast Adaptation", variable=no_contrast_var).pack()

    rgb_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Use RGB", variable=rgb_var).pack()

    noisy_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Noisy image", variable=noisy_var).pack()

    # Run button
    tk.Button(root, text="Run Sharpening", command=start_processing).pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()  # Run CLI mode
    else:
        run_gui()  # Run GUI mode if no args
