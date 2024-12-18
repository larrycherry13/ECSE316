import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import cv2
from matplotlib.colors import LogNorm
from fourier_transforms import fft2d, ifft2d, dft2d
import time

def is_valid_image_file(filename):
    """Check if the file is a valid image file based on its extension."""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return os.path.isfile(filename) and os.path.splitext(filename)[1].lower() in valid_extensions

def parse_input():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process FFT on images.")
    parser.add_argument('-m', '--mode', type=str, default='1',
                        help="Mode of operation: 1=Fast Mode, 2=Denoise, 3=Compress, 4=Runtime Plot")
    parser.add_argument('-i', '--image', type=str, default='moonlanding.png',
                        help="Path to the image file")
    return parser.parse_args()

def fast_mode(image):
    """Display original image and its FFT."""
    fft_result = fft2d(image)
    magnitude_spectrum = np.log(np.abs(fft_result) + 1)  # Log scale

    # Scale down the figure size
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray', norm=LogNorm(), aspect='auto')
    plt.title('Image Fourier Transform')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def denoise(image):
    fft_result = fft2d(image)
    rows, cols = fft_result.shape
    crow, ccol = rows // 2, cols // 2

    cutoff = min(rows, cols) // 9
    Y, X = np.ogrid[:rows, :cols]
    mask = ((X - ccol)**2 + (Y - crow)**2) < cutoff**2
    fft_result[mask] = 0

    denoised_fft = ifft2d(fft_result)
    denoised_image = np.abs(denoised_fft)
    non_zero_count = np.count_nonzero(fft_result)

    print(f"Number of non-zero coefficients: {non_zero_count}")
    print(f"Fraction of original coefficients: {non_zero_count / (rows * cols):.2%}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')
    plt.show()
    

def compress(image):
    """Compress the image by zeroing out smaller Fourier coefficients."""
    # Before compression
    print("Original image mean:", image.mean())

    # Preserve original image range
    image_min = image.min()
    image_max = image.max()
    # Compute 2D FFT of the image
    fft_result = fft2d(image)
    
    # Get magnitude of the FFT
    magnitude = np.abs(fft_result)
    
    # Total number of coefficients
    total_coefficients = magnitude.size
    
    # Define compression levels as fractions of coefficients to zero
    compression_levels = [0.0, 0.05, 0.30, 0.60, 0.9, 0.999]
    
    # Lists to store compressed images and non-zero coefficient counts
    compressed_images = []
    non_zeros = []

    print("Non-Zero Coefficient Analysis:")
    print("----------------------------")

    for lvl in compression_levels:
        # Calculate threshold: keep coefficients above this value
        threshold = np.percentile(magnitude, lvl * 100)
        
        # Create a copy of FFT result to modify
        compressed_fft = fft_result.copy()
        
        # Zero out coefficients below the threshold
        compressed_fft[magnitude < threshold] = 0
        
        # Inverse FFT to get compressed image
        compressed_image = np.real(ifft2d(compressed_fft))

        # Rescale back to original image range
        compressed_image = np.clip(compressed_image, compressed_image.min(), compressed_image.max())
        compressed_image = ((compressed_image - compressed_image.min()) / 
                            (compressed_image.max() - compressed_image.min())) * (image_max - image_min) + image_min
        
        # Store compressed image
        compressed_images.append(compressed_image)
        
        # Count and store non-zero coefficients
        num_non_zeros = np.count_nonzero(compressed_fft)
        non_zeros.append(num_non_zeros)
        
        # Detailed printing
        print(f"Compression Level: {lvl*100:6.1f}%")
        print(f"Non-zero Coefficients:        {num_non_zeros:7d}")
        print(f"Fraction of Coefficients:     {num_non_zeros / total_coefficients:7.2%}")
        print(f"Memory Saved:                 {1 - (num_non_zeros / total_coefficients):7.2%}")
        print("----------------------------")

    # After first FFT and inverse FFT
    print("First compressed image mean:", compressed_images[0].mean())

    # Print pixel value differences
    diff = np.abs(image - compressed_images[0])
    print("Max pixel difference:", diff.max())

    # Print detailed difference statistics
    print("Mean Absolute Error:", np.mean(np.abs(image - compressed_images[0])))
    print("Max Absolute Error:", np.max(np.abs(image - compressed_images[0])))

    # Plot the original and compressed images
    plt.figure(figsize=(15, 10))
    for i, (img, level) in enumerate(zip(compressed_images, compression_levels)):
        plt.subplot(2, 3, i + 1)
        plt.title(f"Compression: {level*100:.1f}%\nNon-zero: {non_zeros[i]}")
        plt.imshow(img, cmap='gray', vmin=image_min, vmax=image_max)
        plt.axis('off')
    plt.show()

def plot():
    sizes = [2**i for i in range(3, 9)]  # From 8x8 to 256x256
    repetitions = 10
    naive_times = []
    fft_times = []
    errors_naive = []
    errors_fft = []

    for size in sizes:
        naive_results = []
        fft_results = []
        for _ in range(repetitions):
            data = np.random.random((size, size))

            # Timing Naive DFT
            start = time.time()
            dft2d(data)
            naive_time = time.time() - start
            naive_results.append(naive_time)

            # Timing FFT
            start = time.time()
            fft2d(data)
            fft_time = time.time() - start
            fft_results.append(fft_time)

        naive_mean = np.mean(naive_results)
        fft_mean = np.mean(fft_results)
        naive_std = np.std(naive_results)
        fft_std = np.std(fft_results)

        naive_times.append(naive_mean)
        fft_times.append(fft_mean)
        errors_naive.append(1.96 * naive_std)  # 95% confidence interval, change to appropriate value for 97%
        errors_fft.append(1.96 * fft_std)      # Same as above

        print(f"Size: {size}x{size}")
        print(f"Naive DFT - Mean: {naive_mean:.5f}s, Std Dev: {naive_std:.5f}s")
        print(f"FFT - Mean: {fft_mean:.5f}s, Std Dev: {fft_std:.5f}s")

    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, naive_times, yerr=errors_naive, label='Naive DFT', fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, markersize=5)
    plt.errorbar(sizes, fft_times, yerr=errors_fft, label='FFT', fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, markersize=5)
    plt.xlabel('Array Size (NxN)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison of DFT and FFT')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

plot()

def main():
    args = parse_input()
    image_path = args.image
    if not is_valid_image_file(image_path):
        print(f"Invalid image file: {image_path}")
        sys.exit(1)

    mode = args.mode
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image from {image_path}")
        sys.exit(1)

    match mode:
        case '1':
            fast_mode(image)
        case '2':
            denoise(image)
        case '3':
            compress(image)
        case '4':
            plot()
        case _:
            print("Invalid mode selected. Please choose from 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
