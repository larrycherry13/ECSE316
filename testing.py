import numpy as np
import pytest
import numpy.testing as npt
from fourier_transforms import (
    naive_dft, 
    inverse_naive_dft, 
    dft2d, 
    idft2d, 
    fft, 
    inverse_fft, 
    fft2d, 
    ifft2d
)

def generate_test_signals():
    """Generate test signals"""
    return {
        # Previous signals
        'sine_wave': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 64)),
        'random_complex': np.random.rand(64) + 1j * np.random.rand(64),
        'test_image': np.random.rand(32, 32),
        
        # New signals
        'small_1d_real': np.array([1, 2, 3, 4], dtype=float),
        'small_2d_real': np.array([[1, 2], [3, 4]], dtype=float),
        'small_1d_complex': np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]),
        'small_2d_complex': np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
        'large_1d_real': np.random.rand(128),
        'large_2d_real': np.random.rand(64, 64),
        'large_1d_complex': np.random.rand(128) + 1j * np.random.rand(128),
        'large_2d_complex': np.random.rand(64, 64) + 1j * np.random.rand(64, 64)
    }

def test_naive_dft_inverse():
    """
    Verify that naive DFT followed by inverse DFT reconstructs the original signal.
    Checks the invertibility property of the Discrete Fourier Transform.
    """
    signals = generate_test_signals()
    
    for name, signal in [('sine_wave', signals['sine_wave']), 
                         ('random_complex', signals['random_complex'])]:
        # Convert to complex to handle both real and complex signals
        signal = signal.astype(complex)
        
        # Perform DFT
        X = naive_dft(signal)
        
        # Perform inverse DFT
        reconstructed = inverse_naive_dft(X)
        
        # Check reconstruction accuracy
        npt.assert_allclose(signal, reconstructed, rtol=1e-10, atol=1e-10, 
                            err_msg=f"Failed to reconstruct {name} signal")

def test_2d_dft_inverse():
    """
    Verify that 2D DFT followed by inverse 2D DFT reconstructs the original image.
    Checks the invertibility property of the 2D Discrete Fourier Transform.
    """
    signals = generate_test_signals()
    image = signals['test_image']
    
    # Perform 2D DFT
    F = dft2d(image)
    
    # Perform inverse 2D DFT
    reconstructed = idft2d(F)
    
    # Check reconstruction accuracy
    npt.assert_allclose(image, reconstructed, rtol=1e-10, atol=1e-10, 
                        err_msg="Failed to reconstruct 2D image using DFT")

def test_fft_inverse():
    """
    Verify that FFT followed by inverse FFT reconstructs the original signal.
    Checks the invertibility property of the Fast Fourier Transform.
    """
    signals = generate_test_signals()
    
    for name, signal in [('sine_wave', signals['sine_wave']), 
                         ('random_complex', signals['random_complex'])]:
        # Convert to complex to handle both real and complex signals
        signal = signal.astype(complex)
        
        # Perform FFT
        X = fft(signal)
        
        # Perform inverse FFT
        reconstructed = inverse_fft(X)
        
        # Check reconstruction accuracy
        npt.assert_allclose(signal, reconstructed, rtol=1e-10, atol=1e-10, 
                            err_msg=f"Failed to reconstruct {name} signal using FFT")

def test_fft_2d_inverse():
    """
    Verify that 2D FFT followed by inverse 2D FFT reconstructs the original image.
    Checks the invertibility property of the 2D Fast Fourier Transform.
    """
    signals = generate_test_signals()
    image = signals['test_image']
    
    # Perform 2D FFT
    F = fft2d(image)
    
    # Perform inverse 2D FFT
    reconstructed = ifft2d(F)
    
    # Check reconstruction accuracy
    npt.assert_allclose(image, reconstructed, rtol=1e-10, atol=1e-10, 
                        err_msg="Failed to reconstruct 2D image using FFT")

def test_fft_numpy_comparison():
    """
    Compare the implementation with NumPy's FFT to validate correctness.
    Checks that our FFT implementation produces results similar to a 
    well-established library implementation.
    """
    signals = generate_test_signals()
    
    for name, signal in [('sine_wave', signals['sine_wave']), 
                         ('random_complex', signals['random_complex'])]:
        # Convert to complex to handle both real and complex signals
        signal = signal.astype(complex)
        
        # Our FFT implementation
        our_fft = fft(signal)
        
        # NumPy's FFT implementation
        numpy_fft = np.fft.fft(signal)
        
        # Compare the results
        npt.assert_allclose(our_fft, numpy_fft, rtol=1e-10, atol=1e-10, 
                            err_msg=f"FFT result differs from NumPy for {name}")


# New tests from the unittest version
def test_large_signal_1d_fft():
    """
    Test FFT on large 1D real and complex signals.
    Verifies reconstruction and comparison with NumPy for larger datasets.
    """
    signals = generate_test_signals()
    
    for name, signal in [
        ('large_1d_real', signals['large_1d_real']),
        ('large_1d_complex', signals['large_1d_complex'])
    ]:
        # Perform FFT
        fft_result = fft(signal)
        
        # Reconstruct signal
        reconstructed = inverse_fft(fft_result)
        
        # Check reconstruction
        npt.assert_allclose(
            signal, reconstructed, 
            rtol=1e-10, atol=1e-10, 
            err_msg=f"Failed to reconstruct {name} signal"
        )
        
        # Compare with NumPy FFT
        numpy_fft = np.fft.fft(signal)
        npt.assert_allclose(
            fft_result, numpy_fft, 
            rtol=1e-10, atol=1e-10, 
            err_msg=f"FFT result differs from NumPy for {name}"
        )

def test_large_signal_2d_fft():
    """
    Test 2D FFT on large real and complex images.
    Verifies reconstruction and comparison with NumPy for larger datasets.
    """
    signals = generate_test_signals()
    
    for name, image in [
        ('large_2d_real', signals['large_2d_real']),
        ('large_2d_complex', signals['large_2d_complex'])
    ]:
        # Perform 2D FFT
        fft_result = fft2d(image)
        
        # Reconstruct image
        reconstructed = ifft2d(fft_result)
        
        # Check reconstruction
        npt.assert_allclose(
            image, reconstructed, 
            rtol=1e-10, atol=1e-10, 
            err_msg=f"Failed to reconstruct {name} image"
        )
        
        # Compare with NumPy 2D FFT
        numpy_fft = np.fft.fft2(image)
        npt.assert_allclose(
            fft_result, numpy_fft, 
            rtol=1e-10, atol=1e-10, 
            err_msg=f"2D FFT result differs from NumPy for {name}"
        )

def test_small_complex_signals():
    """
    Test FFT and inverse FFT on small complex signals.
    Ensures correct handling of complex input.
    """
    signals = generate_test_signals()
    
    for name, signal in [
        ('small_1d_complex', signals['small_1d_complex']),
        ('small_2d_complex', signals['small_2d_complex'])
    ]:
        is_2d = len(signal.shape) > 1
        
        # Perform appropriate FFT
        fft_func = fft2d if is_2d else fft
        ifft_func = ifft2d if is_2d else inverse_fft
        numpy_fft_func = np.fft.fft2 if is_2d else np.fft.fft
        
        # Compute FFT
        fft_result = fft_func(signal)
        
        # Reconstruct signal
        reconstructed = ifft_func(fft_result)
        
        # Check reconstruction of original signal
        npt.assert_allclose(
            signal, reconstructed, 
            rtol=1e-10, atol=1e-10, 
            err_msg=f"Failed to reconstruct complex {name}"
        )
        
        # Compare with NumPy FFT
        numpy_fft = numpy_fft_func(signal)
        npt.assert_allclose(
            fft_result, numpy_fft, 
            rtol=1e-10, atol=1e-10, 
            err_msg=f"Complex FFT result differs from NumPy for {name}"
        )

if __name__ == '__main__':
    pytest.main([__file__])
""" 

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

test_fft_functions() """
