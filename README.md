# ECSE 316 Assignment 2

**Authors**:  
- Laurent Chiricota (ID: 261113415)  
- Samy Sabir (ID: 261119166)  

**Python Version**: 3.11.5
**Requirements**: numpy, cv2, matplotlib, pytest (for testing)

## Project Description
Fourier Transforms and inverses implementations as well as image processing such as denoising, compression

## Usage

To run the program, use the following command structure:

python fft.py [-m mode] [-i image]

- `mode`: (Optional) Processing mode. Default is Fast Mode.
    - [1] (Default) `Fast mode`: Convert image to FFT form and display.
    - [2] `Denoise`: The image is denoised by applying an FFT, truncating high frequencies and then displayed.
    - [3] `Compress`: Compress image and plot.
    - [4] `Plot runtime` graphs for the report.
- `image`: (Optional) Filename of the image for the DFT (default: given image).

## Testing

To run a test file, run in cmd pip install pytest, and use the following command structure:

python [test_filename].py