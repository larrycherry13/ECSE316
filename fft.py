import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import cv2  # For image loading
from matplotlib.colors import LogNorm
from fourier_transforms import fft2d, ifft2d, naive_dft, fft, dft2d # Importing the functions from your module
import time

def is_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return os.path.splitext(filename)[1].lower() in valid_extensions

def parse_input():

    parser = argparse.ArgumentParser(description="Process FFT on images.")
    parser.add_argument('-m', '--mode', type=str, default='1',
                        help="Mode of operation: 1=Fast Mode, 2=Denoise, 3=Compress, 4=Runtime Plot")
    parser.add_argument('-i', '--image', type=str, required=True,
                        help="Path to the image file")
    return parser.parse_args()

def display_images(original, transformed, reconstructed):
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(np.log(np.abs(transformed) + 1), norm=LogNorm(), cmap='gray')
    plt.title('FFT Magnitude Spectrum')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.show()

def fast_mode(image):
    """Display original image and its FFT."""
    fft_result = fft2d(image)
    magnitude_spectrum = np.log(np.abs(fft_result) + 1)  # Log scale

    # Scale down the figure size
    plt.figure(figsize=(8, 4))  # Adjust these numbers as needed to scale down
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
    """denoise the image by removing high frequencies and displaying results"""

    fft_result = fft2d(image)
    # Zero out high frequencies
    rows, cols = fft_result.shape
    crow, ccol = rows // 2, cols // 2
    cutoff = min(rows, cols) // 9  #1/9 of the size
    row_min, row_max = crow - cutoff, crow + cutoff
    col_min, col_max = ccol - cutoff, ccol + cutoff

    fft_result[row_min:row_max, col_min:col_max] = 0

    denoised_fft = ifft2d(fft_result)

    denoised_image = np.abs(denoised_fft)

    print(f"Number of non-zero coefficients: {np.count_nonzero(fft_result)}")
    print(f"Fraction of original coefficients: {np.count_nonzero(fft_result) / (rows * cols):.2%}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image, cmap='gray')
    plt.axis('off')
    plt.show()
    

def compress(image):
    """Compress the image by zeroing out smaller Fourier coefficients."""
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
        # (1 - lvl) ensures we keep more coefficients as compression level increases
        threshold = np.percentile(magnitude, (1 - lvl) * 100)
        
        # Create a copy of FFT result to modify
        compressed_fft = fft_result.copy()
        
        # Zero out coefficients below the threshold
        compressed_fft[magnitude < threshold] = 0
        
        # Inverse FFT to get compressed image
        compressed_image = np.abs(ifft2d(compressed_fft))
        
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

    # Plot the original and compressed images
    plt.figure(figsize=(15, 10))
    for i, (img, level) in enumerate(zip(compressed_images, compression_levels)):
        plt.subplot(2, 3, i + 1)
        plt.title(f"Compression: {level*100:.1f}%\nNon-zero: {non_zeros[i]}")
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()


def measure_runtime(method, size):
    """ Helper function to measure the runtime of a method. """
    x = np.random.random((size, size))
    start_time = time.time()
    method(x)
    return time.time() - start_time


def plot():
    sizes = [2**i for i in range(5, 11)]  # From 32 to 1024
    repetitions = 10  # Number of repetitions for each size

    naive_times = []
    fft_times = []
    errors_naive = []
    errors_fft = []

    for size in sizes:
        naive_results = []
        fft_results = []
        for _ in range(repetitions):
            data = np.random.random((size, size))  # Generate random 2D data

            # Measure runtime for 2D naive DFT
            start_time = time.time()
            dft2d(data)  # Assuming dft2d is the correct function name
            naive_time = time.time() - start_time
            naive_results.append(naive_time)

            # Measure runtime for 2D FFT
            start_time = time.time()
            fft2d(data)  # Assuming fft2d is the correct function name
            fft_time = time.time() - start_time
            fft_results.append(fft_time)

        naive_mean = np.mean(naive_results)
        fft_mean = np.mean(fft_results)
        naive_std = np.std(naive_results)
        fft_std = np.std(fft_results)

        naive_times.append(naive_mean)
        fft_times.append(fft_mean)
        errors_naive.append(naive_std)
        errors_fft.append(fft_std)

        print(f"Size: {size}x{size}")
        print(f"Naive DFT - Mean: {naive_mean:.5f}s, Std: {naive_std:.5f}s")
        print(f"FFT - Mean: {fft_mean:.5f}s, Std: {fft_std:.5f}s")

    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, naive_times, yerr=errors_naive, label='Naive DFT', fmt='-o')
    plt.errorbar(sizes, fft_times, yerr=errors_fft, label='FFT', fmt='-o')
    plt.xlabel('Array Size (NxN)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison of DFT and FFT')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()






def main():
    args = parse_input()
    image_path = args.image
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
