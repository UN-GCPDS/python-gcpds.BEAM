import numpy as np
import time
from detection import Detection
from processing import Processing

class VHF1():
    def __init__(self, sample_rate: int = 20):
        """
        Initialize the VHF1 object with the given parameters and create additional objects for further analysis.

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
        self.frequencies = self.detec.broadcasters('Manizales')


    def parameter(self, data: np.ndarray, fc: int = 98e6, gps: np.ndarray = None) -> dict:
        """Calculate parameters in commercial FM range
        
        This method is for calculating the power and signal-to-noise ratio (SNR) in 
        recorded and detected emissions in the range of 88 to 108 MHz

        Parameters
        ----------
        data : np.ndarray
            Vector of complex numbers representing the I/Q signal captured by the HackRF device.
        fc : int
            Central frequency at which the data was acquired.
        gps: np.ndarray


        Returns
        -------
        Dict
            A dict with that contains:
            - 'time': float : Time at which the trace was captured.
            - 'freq': float : Central frequency of registered or detected emission.
            - 'power': float : Emission power.
            - 'power_max': float : Maximum emission power.
            - 'snr': float : Signal-to-noise ratio (SNR)
        """

        parameters = {
            'time': [],
            'freq': [],
            'power': [],
            'power_max': [],
            'snr': []
        }

        t0 = time.strftime('%X')

        f, Pxx = self.pros.welch(data, self.sample_rate)
        f = f + fc / 1e6

        _, peak_freqs, _ = self.detec.power_based_detection(f, Pxx)
        
        filtered_peak_freqs = []

        for peak_freq in peak_freqs:
            is_redundant = any(np.isclose(peak_freq, center_freq, atol=0.01) for center_freq in self.frequencies)
            if not is_redundant:
                filtered_peak_freqs.append(peak_freq)

        center_freqs = np.sort(np.concatenate((self.frequencies, filtered_peak_freqs)))

        for center_freq in center_freqs:
            index = np.where(np.isclose(f, center_freq, atol=0.01))[0][0]

            lower_index = np.argmin(np.abs(f - (f[index] - 0.125)))
            upper_index = np.argmin(np.abs(f - (f[index] + 0.125)))

            Pxx_range = Pxx[lower_index:upper_index + 1]     

            power = np.mean(Pxx_range)
            power_max = Pxx[index]

            parameters['time'].append(t0)
            parameters['freq'].append(center_freq)
            parameters['snr'].append(10 * np.log10(power / Pxx[0]))
            parameters['power'].append(10 * np.log10(power))
            parameters['power_max'].append(10 * np.log10(power_max))

        return f, Pxx, parameters