import numpy as np

def next_power_of_two(n):
    return 2 ** int(np.ceil(np.log2(n)))

def naive_dft(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def inverse_naive_dft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N


def dft2d(image):
    rows, cols = image.shape
    # compute 1D dft along rows
    row_transform = np.zeros((rows, cols), dtype=complex)
    for n in range(rows):
        row_transform[n, :] = naive_dft(image[n, :])
    
    # compute 1D dft along columns of row-transformed image
    F = np.zeros((rows, cols), dtype=complex)
    for m in range(cols):
        F[:, m] = naive_dft(row_transform[:, m])

    return F

def idft2d(F):
    """naive 2D inverse dft"""
    rows, cols = F.shape
    # compute 1D idft along columns
    col_transform = np.zeros((rows, cols), dtype=complex)
    for m in range(cols):
        col_transform[:, m] = inverse_naive_dft(F[:, m])
    
    # compute 1D idft along rows of column-transformed image
    f = np.zeros((rows, cols), dtype=complex)
    for n in range(rows):
        f[n, :] = inverse_naive_dft(col_transform[n, :])
    
    return np.real(f)

def fft(x):
    """Cooley Tukey recursive fft"""
    # add zeros to make input a power of 2
    N = len(x)
    padded_N = next_power_of_two(N)
    padded_x = np.zeros(padded_N, dtype=complex)
    padded_x[:N] = x

    def _fft_recursive(x):
        N = len(x)
        if N <= 1:  # base case
            return x
        
        # split into even and odd indices
        even = _fft_recursive(x[0::2])
        odd = _fft_recursive(x[1::2])

        # compute twiddle factors
        factors = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        
        # combine even and odd parts
        first_half = even + factors * odd
        second_half = even - factors * odd
        
        return np.concatenate([first_half, second_half])

    # compute fft and trim back to original length
    result = _fft_recursive(padded_x)
    return result[:N]


def inverse_fft(x):
    """Cooley-Tukey recursive inverse fft"""
    # add zeros to make input a power of 2
    N = len(x)
    padded_N = next_power_of_two(N)
    padded_x = np.zeros(padded_N, dtype=complex)
    padded_x[:N] = x

    def _ifft_recursive(x):
        N = len(x)
        if N <= 1:  # base case
            return x
        
        # split into even and odd indices
        even = _ifft_recursive(x[0::2])
        odd = _ifft_recursive(x[1::2])

        # compute twiddle factors
        factors = np.exp(2j * np.pi * np.arange(N // 2) / N)
        
        # combine even and odd parts
        first_half = even + factors * odd
        second_half = even - factors * odd
        
        return np.concatenate([first_half, second_half])

    # compute ifft and trim back to original length
    result = _ifft_recursive(padded_x)
    return result[:N] / padded_N

def fft2d(image):
    """2D fft"""
    rows, cols = image.shape
    # compute 1D fft along rows
    row_transform = np.zeros((rows, cols), dtype=complex)
    for n in range(rows):
        row_transform[n, :] = fft(image[n, :])
    
    # compute 1D fft along columns of row-transformed image
    F = np.zeros((rows, cols), dtype=complex)
    for m in range(cols):
        F[:, m] = fft(row_transform[:, m])
    
    return F

def ifft2d(F):
    """2D inverse fft"""
    rows, cols = F.shape
    # compute 1D ifft along columns
    col_transform = np.zeros((rows, cols), dtype=complex)
    for m in range(cols):
        col_transform[:, m] = inverse_fft(F[:, m])
    
    # Compute 1D ifft along rows of column-transformed image
    f = np.zeros((rows, cols), dtype=complex)
    for n in range(rows):
        f[n, :] = inverse_fft(col_transform[n, :])
    
    return np.real(f)

def fft_shift(fft_data):
    """shift the zero-frequency component to the center of the spectrum. """
    rows, cols = fft_data.shape
    return np.roll(np.roll(fft_data, rows // 2, axis=0), cols // 2, axis=1)

def ifft_shift(fft_data):
    """shift back the zero-frequency component to the original position. """
    rows, cols = fft_data.shape
    return np.roll(np.roll(fft_data, -rows // 2, axis=0), -cols // 2, axis=1)