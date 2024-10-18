import numpy as np
import time
from detection import Detection
from processing import Processing

class VHF2():
    def __init__(self, sample_rate: int = 20):
        """
        Initialize the VHF2 object with the given parameters and create additional objects for further analysis.

        Parameters:
        sample_rate (int): The sample rate used for signal analysis, in Hz. The default value is 20 MHz.

        Attributes:
        detec (Detection): Instance of the Detection class for frequency detection.
        pros (Processing): Instance of the Processing class for signal processing.
        sample_rate (int): The sample rate for analysis.
        frequencies (list): List of channels frequencies in range 137MHz to 144MHz.
        bandwidth (list): List of the bandwidth of each channel.
        """
        self.detec = Detection()
        self.pros = Processing()
        self.sample_rate = sample_rate
        self.frequencies, self.bandwidth = np.load()

    def parameter(self, data: np.ndarray, nper: int = 32768, fc: float = 140.5e6, gps: np.ndarray = None) -> dict:
        """Calculate parameters in Fixed and Movil service (137MHz to 144MHz)
        
        This method is for calculating the power corresponding to each channel in the 
        given range.

        Parameters
        ----------
        data : np.ndarray
            Vector of complex numbers representing the I/Q signal captured by the HackRF device.
        nper : int
            Length of each segment.
        fc : int
            Central frequency at which the data was acquired.
        gps : np.ndarray
            
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

        f, Pxx = self.pros.welch(data, nper=nper, fs=self.sample_rate)
        f = (f + fc) / 1e6

        for center_freq in self.frequencies:
            index = np.where(np.isclose(f, center_freq, atol=0.01))[0][0]

            lower_index = np.argmin(np.abs(f - (f[index] - self.bandwidth[center_freq]/2)))
            upper_index = np.argmin(np.abs(f - (f[index] + self.bandwidth[center_freq]/2)))

            Pxx_range = Pxx[lower_index:upper_index + 1] 

            power = np.mean(10 * np.log10(Pxx_range))
            power_max = Pxx[index]

            parameters['time'].append(t0)
            parameters['freq'].append(center_freq)
            parameters['snr'].append(10 * np.log10(power / Pxx[0]))
            parameters['power'].append(10 * np.log10(power))
            parameters['power_max'].append(10 * np.log10(power_max))

        return f, Pxx, parameters