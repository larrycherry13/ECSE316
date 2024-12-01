import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import cv2  # For image loading
from matplotlib.colors import LogNorm
from fourier_transforms import fft2d, ifft2d, naive_dft, fft, dft2d # Importing the functions from your module
import time

def denoise_low_pass(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return
    fft_result = fft2d(image)
    rows, cols = fft_result.shape
    crow, ccol = rows // 2, cols // 2

    cutoff = min(rows, cols) // 9
    Y, X = np.ogrid[:rows, :cols]
    mask = ((X - ccol)**2 + (Y - crow)**2) > cutoff**2
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
    plt.title("Denoised Image (Low-Pass Filter)")
    plt.axis('off')
    plt.show()
def denoise_high_pass(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return
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
    plt.title("Denoised Image (High-Pass Filter)")
    plt.axis('off')
    plt.show()

def denoise_band_pass(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return
    fft_result = fft2d(image)
    rows, cols = fft_result.shape
    crow, ccol = rows // 2, cols // 2

    low_cutoff = min(rows, cols) // 16
    high_cutoff = min(rows, cols) // 8
    Y, X = np.ogrid[:rows, :cols]
    mask_low = ((X - ccol)**2 + (Y - crow)**2) < low_cutoff**2
    mask_high = ((X - ccol)**2 + (Y - crow)**2) > high_cutoff**2
    mask = mask_low | mask_high
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
    plt.title("Denoised Image (Band-Pass Filter)")
    plt.axis('off')
    plt.show()
# Example usage
def denoise_thresholding(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return
    
    fft_result = fft2d(image)
    rows, cols = fft_result.shape  # Define rows and cols here

    magnitude = np.abs(fft_result)
    threshold = np.percentile(magnitude, 95)  # keep top 5% of frequencies
    fft_result[magnitude < threshold] = 0

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
    plt.title("Denoised Image (Thresholding Filter)")
    plt.axis('off')
    plt.show()


denoise_thresholding("moonlanding (1).png")