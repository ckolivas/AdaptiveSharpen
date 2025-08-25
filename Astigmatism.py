#!/usr/bin/python3
import argparse  # NEW: For command-line arguments
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

    # Step 2: Estimate background and noise with sigma-clipping
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    data -= median  # Subtract background
    print(f"Background median: {median}, Std: {std}, Max value: {np.max(data)}")  # Debug

    # Step 3: Detect peaks (stars) - use absolute threshold for robustness
    threshold_abs = 10 * std  # Tune multiplier (3-10) if needed; lower for more/fainter detections
    coordinates = peak_local_max(data, min_distance=4, threshold_abs=threshold_abs)
    print(f"Number of detected peaks: {len(coordinates)}")  # Debug

    # Fallback: If no peaks, use the brightest point (e.g., Triton)
    if len(coordinates) == 0:
        y, x = np.unravel_index(np.argmax(data), data.shape)
        coordinates = np.array([[y, x]])
        print("Fallback: Using brightest pixel as single source")  # Debug

    # Optional: Filter for top N brightest peaks
    if len(coordinates) > 50:
        fluxes = data[coordinates[:, 0], coordinates[:, 1]]
        sorted_idx = np.argsort(fluxes)[-50:]  # Top 50 by peak value
        coordinates = coordinates[sorted_idx]

    # Step 4: Extract and center cutouts
    cutout_size = 21  # Odd number, ~7x FWHM (adjust to 15-30 based on wings)
    half_size = cutout_size // 2
    cutouts = []
    for y, x in coordinates:
        # Check bounds
        if (y - half_size < 0 or y + half_size >= data.shape[0] or
            x - half_size < 0 or x + half_size >= data.shape[1]):
            continue
        cutout = data[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]

        # Step 5: Center sub-pixel using center of mass
        cy, cx = center_of_mass(cutout)
        shift_y = half_size - cy
        shift_x = half_size - cx
        centered_cutout = shift(cutout, (shift_y, shift_x), mode='constant', cval=0)

        # Normalize by integrated flux
        flux = np.sum(centered_cutout)
        if flux > 0:
            centered_cutout /= flux
            cutouts.append(centered_cutout)

    print(f"Number of valid cutouts: {len(cutouts)}")  # Debug

    # Step 6: Average the cutouts to get the PSF
    if cutouts:
        custom_psf = np.mean(cutouts, axis=0)
        # Normalize PSF to sum=1 for deconvolution
        custom_psf /= np.sum(custom_psf)
    else:
        raise ValueError("No valid cutouts found; check detection parameters and image bounds.")

    # NEW: Save the PSF to the specified .npy file
    np.save(args.output, custom_psf)
    print(f"PSF saved to: {args.output} (shape: {custom_psf.shape})")

if __name__ == '__main__':
    main()
