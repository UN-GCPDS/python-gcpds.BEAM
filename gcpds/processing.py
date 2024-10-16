import numpy as np
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper
from typing import Literal
import pywt
from scipy.signal import resample_poly


class Processing:
    """
    This class provides various methods for processing signals,
    including Fourier Transform, spectral estimation using Welch's method,
    Multi-taper method, and Wavelet transform.
    """

    def convert_to_real(self, signal: np.ndarray, method: Literal['inter', 'sincos'], sample_rate: int=20e6, time_to_read: float=0.01, center_freq: int=None,) -> np.ndarray:
        """
        Convert the complex signal to a real signal by interpolating and frequency shifting or using the sin cosin signal.

        Parameters
        ----------
        signal : np.ndarray
            An array representing the complex signal where the real part is the in-phase component (I)
            and the imaginary part is the quadrature component (Q).
        method : Literal['inter', 'sincos']
            Method for conversion.
        sample_rate : int, optional
            The sampling rate of the signal. Defaults to 20e6.
        center_freq : int, optional
            Center frequency. Defaults to None.

        Returns
        -------
        np.ndarray
            A signal converted in real signal
        """
        match method:
            case 'inter':
                signal_iq_interp_real = resample_poly(signal.real, up=2, down=1, padtype='line')
                signal_iq_interp_imag = resample_poly(signal.imag, up=2, down=1, padtype='line')
                signal_iq_interp = signal_iq_interp_real + 1j * signal_iq_interp_imag

                freq_shift = sample_rate/2
                fs_real = sample_rate * 2
                time_vector = np.arange(len(signal_iq_interp))
                complex_sine = np.exp(1j*2*np.pi* (freq_shift/fs_real) * time_vector)
                signal_shifted = signal_iq_interp * complex_sine

                signal = signal_shifted.real

            case 'sincos':
                t = np.linspace(0, time_to_read, len(signal))
                signal = signal.real * np.cos(2 * np.pi * center_freq * t) - signal.imag * np.sin(2 * np.pi * center_freq * t)
            
        return signal

    
    # ----------------------------------------------------------------------
    def fft(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the Fast Fourier Transform of the given signal.

        Parameters
        ----------
        signal : np.ndarray
            An array representing the signal to be transformed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).

        Returns
        -------
        numpy.ndarray
            The transformed signal in the frequency domain.

        Raises
        ------
        ValueError
            If the input signal is not a numpy array.

        Notes
        -----
        This method utilizes NumPy's FFT function to transform the signal from
        the time domain to the frequency domain.
        """
        if not isinstance(signal, np.ndarray):
            raise ValueError("Input signal must be a numpy array")

        N = len(signal)
        fft_result = np.fft.fft(signal)
        
        #fft = fft_result[:N//2]
        fft = np.abs(fft_result)
        fft = np.fft.fftshift(fft)

        return fft

    # ----------------------------------------------------------------------
    def welch(self, signal: np.ndarray, fs: float = 20e6) -> np.ndarray:
        """
        Estimate the power spectral density of the given signal using Welch's method.

        Parameters
        ----------
        signal : np.ndarray
            An array representing the signal to be analyzed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).
        fs : float, optional
            The sampling frequency of the signal. Default is 20M.

        Returns
        -------
        numpy.ndarray
            The estimated power spectral density of the signal.

        Raises
        ------
        ValueError
            If the input signal is not a numpy array.

        Notes
        -----
        This method utilizes Welch's algorithm, implemented in NumPy, to estimate
        the power spectral density of the signal. The signal is divided into
        overlapping segments, windowed, and then averaged to reduce variance.
        """

        """
        nperseg: int, optional
            Specifies the number of points in each segment. During the calculation of the 
            PSD using the Welch method, the signal is divided into segments, 
            and the PSD is calculated for each segment

            Length of each segment. Defaults to None, but if window is str or tuple, 
            is set to 256, and if window is array_like, is set to the length of the window.

        noverlap: int, optional
            Specifies the number of overlap points between consecutive segments. 
            By overlaying the segments, a smoother and less noisy estimate of the PSD is obtained

            Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
        """

        if not isinstance(signal, np.ndarray):
            raise ValueError("Input signal must be un numpy array")

        # window=('kaiser', 20)
        f, Pxx = welch(signal, fs=20e6, nperseg=1024, window=np.hanning(1024), noverlap=0)

        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)
        return f, Pxx

    # ----------------------------------------------------------------------
    def multi_taper(self, signal: np.ndarray, fs: float = 20e6) -> np.ndarray:
        """
        Estimate the power spectral density of the given signal using the Multi-taper method.

        Parameters
        ----------
        signal : np.ndarray
            An array representing the signal to be analyzed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).
        fs : float, optional
            The sampling frequency of the signal. Default is 20e6.

        Returns
        -------
        numpy.ndarray
            The estimated power spectral density of the signal.

        Raises
        ------
        ValueError
            If the input signal is not a numpy array.

        Notes
        -----
        This method utilizes the Multi-taper method to estimate the power spectral
        density of the signal. The method is often favored for its ability to
        reduce spectral leakage and provide better frequency resolution. Multiple
        orthogonal windowing functions (tapers) are used to obtain independent
        estimates of the power spectrum, which are then averaged.
        """

        """
        The psd_array_multitaper from the mne.time_frequency module is being used 
        due to its robust implementation of the multi-taper method for estimating 
        power spectral density (PSD). Although mne is primarily designed for the 
        analysis of neurophysiological signals, the multi-taper method is a general 
        technique applicable to any type of signal, including radio frequency (RF) signal.
        This specific implementation is well known for its ability to reduce spectral 
        leakage and provide better frequency resolution, which is critical for RF signal analysis.
        """

        if not isinstance(signal, np.ndarray):
            raise ValueError("Input signal must be a numpy array")        

        psd, freqs = psd_array_multitaper(signal, sfreq=fs)
        return freqs, psd

    # ----------------------------------------------------------------------
    def wavelet(self, signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    

        """
        Perform wavelet transform on the given signal.

        Parameters
        ----------
        signal : np.ndarray
            An array representing the signal to be transformed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).
        scales : np.ndarray
            An array of scales to use in the wavelet transform.

        Returns
        -------
        numpy.ndarray
            The wavelet-transformed signal in the time-frequency domain.

        Raises
        ------
        ValueError
            If the input signal is not a numpy array.

        Notes
        -----
        This method utilizes wavelet transform techniques to analyze the signal
        in both time and frequency domains. Wavelet transforms are highly effective
        for analyzing non-stationary signals.
        """
        if not isinstance(signal, np.ndarray):
            raise ValueError("Input signal must be a numpy array")
    

        coeffs, _ = pywt.cwt(signal, scales, 'cmor')
        return coeffs
