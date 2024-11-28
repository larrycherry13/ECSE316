import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import cv2  # For image loading
from matplotlib.colors import LogNorm
from fourier_transforms import fft2d, ifft2d, fft_shift, ifft_shift  # Importing the functions from your module

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
