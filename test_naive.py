import numpy as np
from fourier_transforms import naive_dft


def test_naive_dft():
    # Test input: Single frequency sinusoid
    t = np.linspace(0, 1, 8, endpoint=False)
    single_freq_signal = np.sin(2 * np.pi * 1 * t)

    # Calculate DFT using the naive method
    naive_dft_output = naive_dft(single_freq_signal)

    # Calculate DFT using NumPy's FFT to compare
    numpy_fft_output = np.fft.fft(single_freq_signal)

    # Check if the outputs are close enough
    assert np.allclose(naive_dft_output, numpy_fft_output), "Naive DFT does not match NumPy's FFT"

    # Test with an array of zeros
    zeros_signal = np.zeros(8)
    zeros_dft_output = naive_dft(zeros_signal)
    expected_zeros_output = np.zeros(8, dtype=complex)
    assert np.allclose(zeros_dft_output, expected_zeros_output), "DFT of zero signal should be zeros"

    # Test with a single element
    single_element_signal = np.array([1])
    single_element_dft_output = naive_dft(single_element_signal)
    expected_single_element_output = np.array([1], dtype=complex)
    assert np.allclose(single_element_dft_output, expected_single_element_output), "DFT of a single element array should be the same element"

    # Test with an array of ones
    ones_signal = np.ones(8)
    ones_dft_output = naive_dft(ones_signal)
    # The DFT of ones should only have the first component non-zero equal to the sum of the ones
    expected_ones_output = np.zeros(8, dtype=complex)
    expected_ones_output[0] = 8
    assert np.allclose(ones_dft_output, expected_ones_output), "DFT of ones signal incorrect"

    return (print("All tests passed!"))

# Call the test function
test_naive_dft()
