import numpy as np
import matplotlib.pyplot as plt
from fourier_transforms import fft2d, ifft2d
import cv2

def compress_by_magnitude(fft_result, keep_percent):
    """Retain the top `keep_percent` coefficients by magnitude"""
    threshold = np.percentile(np.abs(fft_result), (1 - keep_percent) * 100)
    compressed_fft = fft_result.copy()
    compressed_fft[np.abs(compressed_fft) < threshold] = 0
    return compressed_fft

def compress_low_freq_and_top_magnitude(fft_result, low_freq_size, keep_percent):
    """Retain all low-frequency coefficients and top `keep_percent` high-frequency coefficients"""
    rows, cols = fft_result.shape
    crow, ccol = rows // 2, cols // 2
    row_min, row_max = crow - low_freq_size, crow + low_freq_size
    col_min, col_max = ccol - low_freq_size, ccol + low_freq_size

    # Preserve low frequencies
    compressed_fft = fft_result.copy()
    low_freq_region = np.zeros_like(fft_result, dtype=bool)
    low_freq_region[row_min:row_max, col_min:col_max] = True

    # Retain top `keep_percent` high-frequency coefficients
    threshold = np.percentile(np.abs(fft_result[~low_freq_region]), (1 - keep_percent) * 100)
    compressed_fft[~low_freq_region & (np.abs(compressed_fft) < threshold)] = 0

    return compressed_fft

def compress_band_filtering(fft_result, target_compression_ratio):
    """
    Retain coefficients within a frequency band to achieve a target compression ratio.
    Dynamically adjusts inner and outer cutoffs to match the desired ratio.
    """
    rows, cols = fft_result.shape
    crow, ccol = rows // 2, cols // 2

    # Compute distances from the center
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)

    # Initialize inner and outer cutoff values
    total_coeffs = rows * cols
    inner_cutoff = 0
    outer_cutoff = max(rows, cols)  # Start with the largest possible cutoff

    # Iteratively adjust the cutoffs to reach the target compression ratio
    while True:
        # Create a mask for the band
        band_mask = (distance >= inner_cutoff) & (distance <= outer_cutoff)

        # Calculate the compression ratio for the current band
        num_nonzero = np.count_nonzero(band_mask)
        compression_ratio = 1 - (num_nonzero / total_coeffs)

        if compression_ratio >= target_compression_ratio:
            break  # Stop when the target compression ratio is reached

        # Narrow the band by increasing inner cutoff and/or decreasing outer cutoff
        inner_cutoff += 1
        outer_cutoff -= 1

        # Avoid invalid cutoffs
        if inner_cutoff >= outer_cutoff:
            break

    # Apply the final band mask to the FFT result
    compressed_fft = np.zeros_like(fft_result, dtype=complex)
    compressed_fft[band_mask] = fft_result[band_mask]

    return compressed_fft

def evaluate_compression(original_image, fft_result, compression_function, **kwargs):
    """Apply a compression function, reconstruct the image, and compute quality metrics"""
    compressed_fft = compression_function(fft_result, **kwargs)
    reconstructed_image = np.abs(ifft2d(compressed_fft))

    # Compute compression ratio
    num_nonzero = np.count_nonzero(compressed_fft)
    total_coeffs = fft_result.size
    compression_ratio = 1 - (num_nonzero / total_coeffs)

    # Compute MSE
    mse = np.mean((original_image - reconstructed_image) ** 2)

    return reconstructed_image, compression_ratio, num_nonzero, mse

def run_compression_experiments(image):
    """Run compression experiments with different methods and visualize results"""
    fft_result = fft2d(image)

    # Methods and parameters
    methods = [
        ("Magnitude Threshold", compress_by_magnitude, {"keep_percent": 0.1}),
        ("Low Freq + Top Magnitude", compress_low_freq_and_top_magnitude, {"low_freq_size": 8, "keep_percent": 0.1}),
        ("Band Filtering", compress_band_filtering, {"target_compression_ratio": 0.9}),
    ]

    results = []

    for method_name, method, params in methods:
        reconstructed_image, compression_ratio, num_nonzero, mse = evaluate_compression(image, fft_result, method, **params)
        results.append((method_name, reconstructed_image, compression_ratio, num_nonzero, mse))

    # Display results
    plt.figure(figsize=(15, 10))

    # Display original image in the first subplot
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    for i, (method_name, reconstructed_image, compression_ratio, num_nonzero, mse) in enumerate(results):
        plt.subplot(2, 2, i + 2)
        plt.imshow(reconstructed_image, cmap="gray")
        plt.title(f"{method_name}\nCompression: {compression_ratio:.2%}\nNumber of Coefficients: {num_nonzero}\nMSE: {mse:.2f}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return results

if __name__ == "__main__":
    # Load moonlanding.png image
    test_image = cv2.imread("moonlanding.png", cv2.IMREAD_GRAYSCALE)

    # Run compression experiments
    results = run_compression_experiments(test_image)
