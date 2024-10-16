import numpy as np
import time
import matplotlib.pyplot as plt
from detection import Detection
from processing import Processing

class VHF2():
    def __init__(self, sample_rate: int = 8e6):
        """
        Initialize the VHF2 object with the given parameters and create additional objects for further analysis.

        Parameters:
        sample_rate (int): The sample rate used for signal analysis, in Hz. The default value is 20 MHz.

        Attributes:
        detec (Detection): Instance of the Detection class for frequency detection.
        pros (Processing): Instance of the Processing class for signal processing.
        sample_rate (int): The sample rate for analysis.
        frequencies (list): List of broadcasters' frequencies obtained for the 'Manizales' region.
        """
        self.detec = Detection()
        self.pros = Processing()
        self.sample_rate = sample_rate

    def parameter(self, data: np.ndarray, fc: float = 2617.5e6) -> dict:
        """Calculate parameters in commercial FM range
        
        This method is for calculating the power and signal-to-noise ratio (SNR) in 
        recorded and detected emissions in the range of 88 to 108 MHz

        Parameters
        ----------
        data : np.ndarray
            Vector of complex numbers representing the I/Q signal captured by the HackRF device.
        fc : int
            Central frequency at which the data was acquired.

        Returns
        -------
        Dict
            A dict with that contains:
            - 'time': float : Time at which the trace was captured.
            - 'freq': float : Central frequency of registered or detected emission.
            - 'power': float : Emission power.
            - 'snr': float : Signal-to-noise ratio (SNR)
        """
        
        parameters = {
            'time': [],
            'freq': [],
            'power': [],
            'snr': []
        }

        t0 = time.strftime('%X')

        f, Pxx = self.pros.welch(data, self.sample_rate)
        f = (f + fc) / 1e6

        peak_powers, peak_freqs, threshold = self.detec.power_based_detection(f, Pxx)

        for center_freq in peak_freqs:
            index = np.where(np.isclose(f, center_freq, atol=0.01))[0][0]

            lower_index = np.argmin(np.abs(f - (f[index] - 2.5)))
            upper_index = np.argmin(np.abs(f - (f[index] + 2.5)))

            freq_range = f[lower_index:upper_index + 1]
            Pxx_range = Pxx[lower_index:upper_index + 1]

            # plt.semilogy(freq_range, Pxx_range) 

            power = np.trapz(10 * np.log10(Pxx_range), freq_range)

            parameters['time'].append(t0)
            parameters['freq'].append(center_freq)
            parameters['snr'].append(10 * np.log10(power / Pxx[0]))
            parameters['power'].append(10 * np.log10(power))

        # plt.show()

        return f, Pxx, parameters