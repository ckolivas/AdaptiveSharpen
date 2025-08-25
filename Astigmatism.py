#!/usr/bin/python3
import argparse
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import shift, center_of_mass
from skimage.feature import peak_local_max

def main():
    parser = argparse.ArgumentParser(description='Extract custom PSF from a FITS image for deconvolution.')
    parser.add_argument('--input', type=str, default='custom.fits', help='Input FITS file path (default: custom.fits)')
    parser.add_argument('--output', type=str, default='custom_psf.npy', help='Output PSF .npy file path (default: custom_psf.npy)')
    args = parser.parse_args()

    # Step 1: Load the stacked image
    with fits.open(args.input) as hdul:
        data = hdul[0].data.astype(float)  # Ensure float for processing

    # Handle common FITS formats: if channels first (3, h, w), move to (h, w, 3)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)

    # Determine if RGB or mono
    is_rgb = (data.ndim == 3 and data.shape[-1] == 3)
    if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] != 3):
        raise ValueError(f"Unsupported data shape: {data.shape}. Expected 2D or (h, w, 3).")

    # Step 2: Estimate background and noise with sigma-clipping (per channel if RGB)
    if not is_rgb:
        mean, median, std = sigma_clipped_stats(data)
        data -= median
        medians = [median]
        stds = [std]
    else:
        medians = []
        stds = []
        for ch in range(3):
            mean_ch, med_ch, std_ch = sigma_clipped_stats(data[..., ch])
            data[..., ch] -= med_ch
            medians.append(med_ch)
            stds.append(std_ch)
    median_avg = np.mean(medians)
    std_avg = np.mean(stds)
    print(f"Background median (avg): {median_avg}, Std (avg): {std_avg}, Max value: {np.max(data)}")  # Debug

    # Compute luminance for detection
    lum_for_detection = np.mean(data, axis=-1) if is_rgb else data

    # Step 3: Detect peaks (stars) - use absolute threshold for robustness
    threshold_abs = 5 * std_avg  # Tune multiplier (3-10); uses average std
    coordinates = peak_local_max(lum_for_detection, min_distance=4, threshold_abs=threshold_abs)
    print(f"Number of detected peaks: {len(coordinates)}")  # Debug

    # Fallback: If no peaks, use the brightest point
    if len(coordinates) == 0:
        y, x = np.unravel_index(np.argmax(lum_for_detection), lum_for_detection.shape)
        coordinates = np.array([[y, x]])
        print("Fallback: Using brightest pixel as single source")  # Debug

    # Optional: Filter for top N brightest peaks
    if len(coordinates) > 50:
        fluxes = lum_for_detection[coordinates[:, 0], coordinates[:, 1]]
        sorted_idx = np.argsort(fluxes)[-50:]
        coordinates = coordinates[sorted_idx]

    # Step 4: Extract and center cutouts
    cutout_size = 21  # Odd number, ~7x FWHM (adjust to 15-30 based on wings)
    half_size = cutout_size // 2
    cutouts = []
    for y, x in coordinates:
        y_start = max(0, y - half_size)
        y_end = min(data.shape[0], y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(data.shape[1], x + half_size + 1)

        # Skip if cutout too small
        if (y_end - y_start < cutout_size) or (x_end - x_start < cutout_size):
            continue

        if not is_rgb:
            cutout = data[y_start:y_end, x_start:x_end]
            # Pad if necessary (though bounds check minimizes this)
            if cutout.shape[0] < cutout_size or cutout.shape[1] < cutout_size:
                pad_y = cutout_size - cutout.shape[0]
                pad_x = cutout_size - cutout.shape[1]
                cutout = np.pad(cutout, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

            cy, cx = center_of_mass(cutout)
            shift_y = half_size - cy
            shift_x = half_size - cx
            centered_cutout = shift(cutout, (shift_y, shift_x), mode='constant', cval=0)
            flux = np.sum(centered_cutout)
            if flux > 0:
                centered_cutout /= flux
                cutouts.append(centered_cutout)
        else:
            centered_channels = []
            for ch in range(3):
                cutout_ch = data[y_start:y_end, x_start:x_end, ch]
                if cutout_ch.shape[0] < cutout_size or cutout_ch.shape[1] < cutout_size:
                    pad_y = cutout_size - cutout_ch.shape[0]
                    pad_x = cutout_size - cutout_ch.shape[1]
                    cutout_ch = np.pad(cutout_ch, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

                cy, cx = center_of_mass(cutout_ch)
                shift_y = half_size - cy
                shift_x = half_size - cx
                centered_ch = shift(cutout_ch, (shift_y, shift_x), mode='constant', cval=0)
                flux_ch = np.sum(centered_ch)
                if flux_ch > 0:
                    centered_ch /= flux_ch
                centered_channels.append(centered_ch)
            if all(np.sum(ch) > 0 for ch in centered_channels):
                cutouts.append(np.stack(centered_channels, axis=-1))

    print(f"Number of valid cutouts: {len(cutouts)}")  # Debug

    # Step 6: Average the cutouts to get the PSF
    if cutouts:
        custom_psf = np.mean(cutouts, axis=0)
        # Normalize per channel if RGB, or overall if mono
        if custom_psf.ndim == 3:
            for ch in range(3):
                psf_sum = np.sum(custom_psf[..., ch])
                if psf_sum != 0:
                    custom_psf[..., ch] /= psf_sum
        else:
            custom_psf /= np.sum(custom_psf)
    else:
        raise ValueError("No valid cutouts found; check detection parameters and image bounds.")

    # Save the PSF to the specified .npy file
    np.save(args.output, custom_psf)
    print(f"PSF saved to: {args.output} (shape: {custom_psf.shape})")

if __name__ == '__main__':
    main()
