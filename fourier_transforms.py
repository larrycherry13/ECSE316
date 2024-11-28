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

def fft(x):
    N = len(x)
    if N & (N - 1):  # Ensure N is a power of two
        padded_N = next_power_of_two(N)
        x = np.resize(x, padded_N)
    return rec_fft(x)

def rec_fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = rec_fft(x[0::2])
    odd = rec_fft(x[1::2])
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + T, even - T])


def inverse_fft(X):
    N = len(X)
    if N <= 1:
        return X
    even = inverse_fft(X[0::2])
    odd = inverse_fft(X[1::2])
    T = [np.exp(2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [(even[k] + T[k]) / 2 for k in range(N // 2)] + [(even[k] - T[k]) / 2 for k in range(N // 2)]

def fft2d(image):
    return np.apply_along_axis(fft, axis=1, arr=np.apply_along_axis(fft, axis=0, arr=image))

def ifft2d(F):
    return np.apply_along_axis(inverse_fft, axis=1, arr=np.apply_along_axis(inverse_fft, axis=0, arr=F))

