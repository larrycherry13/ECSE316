import numpy as np
import matplotlib.pyplot as plt
from fourier_transforms import fft, inverse_fft

def visualize_1d_fft():
    # Generate a sinusoidal signal
    t = np.linspace(0, 1, 256, endpoint=False)
    frequency = 8  # frequency of the sine wave
    signal = np.sin(2 * np.pi * frequency * t) + 0.5 * np.sin(2 * np.pi * 2 * frequency * t)

    # Perform FFT
    signal_fft = fft(signal)

    # Perform inverse FFT
    signal_ifft = inverse_fft(signal_fft)

    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot original signal
    axs[0].plot(t, signal, label='Original Signal')
    axs[0].set_title('Time Domain - Original Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()

    # Plot FFT magnitudes
    frequencies = np.fft.fftfreq(len(signal), d=t[1] - t[0])
    axs[1].stem(frequencies, np.abs(signal_fft), 'b', markerfmt=" ", basefmt="-b")
    axs[1].set_title('Frequency Domain - Magnitude Spectrum')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_xlim(0, 50)  # Limit this to lower frequencies for better visibility

    # Plot reconstructed signal from IFFT
    axs[2].plot(t, signal_ifft, label='Reconstructed Signal', color='green')
    axs[2].set_title('Time Domain - Reconstructed Signal from IFFT')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

visualize_1d_fft()
