import numpy as np
from fourier_transforms import naive_dft, inverse_naive_dft, fft, inverse_fft, fft_2d, inverse_fft_2d

def test_fft_functions():
    # Simple impulse image
    image = np.zeros((8, 8))
    image[1, 1] = 1  # Simple impulse
    
    # Apply custom and NumPy 2D FFT
    my_fft_2d = fft_2d(image)
    numpy_fft_2d = np.fft.fft2(image)
    
    # Print and compare outputs
    if not np.allclose(my_fft_2d, numpy_fft_2d):
        print("Mismatch found:")
        print("Custom FFT 2D:\n", my_fft_2d)
        print("NumPy FFT 2D:\n", numpy_fft_2d)
        # Check the transformation by rows
        fft_rows = np.array([fft(row) for row in image])
        numpy_fft_rows = np.fft.fft(image, axis=0)
        print("Custom FFT Rows:\n", fft_rows)
        print("NumPy FFT Rows:\n", numpy_fft_rows)
        # Check the transformation by columns
        fft_cols = np.array([fft(col) for col in fft_rows.T]).T
        numpy_fft_cols = np.fft.fft(numpy_fft_rows.T, axis=0).T
        print("Custom FFT Cols:\n", fft_cols)
        print("NumPy FFT Cols:\n", numpy_fft_cols)
    
    assert np.allclose(my_fft_2d, numpy_fft_2d), "2D FFT mismatch with NumPy on simple impulse"
    print("All tests passed!")


def test_with_sinusoidal_signal():
    t = np.linspace(0, 1, 256, endpoint=False)
    frequency = 5  # frequency of the sine wave
    signal = np.sin(2 * np.pi * frequency * t)
    
    my_fft = fft(signal)
    numpy_fft = np.fft.fft(signal)
    
    if not np.allclose(my_fft, numpy_fft):
        print("Mismatch found in sinusoidal test:")
        print("Custom FFT:\n", my_fft)
        print("NumPy FFT:\n", numpy_fft)
    else:
        print("Sinusoidal test passed!")


test_with_sinusoidal_signal()

test_fft_functions()
