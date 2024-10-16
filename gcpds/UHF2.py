import numpy as np
import time
from detection import Detection
from processing import Processing

class VHF1():
    def __init__(self, sample_rate: int = 8e6, bits_per_symbol: int = 6):
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

        self.bits_per_symbol = bits_per_symbol
        self.M = 2 ** bits_per_symbol
        self.constellation = self.generate_constellation()
        self.symbol_mapping = self.generate_symbol_mapping()

    def generate_constellation(self):
        """
        Generates a normalized QAM constellation based on the specified bits per symbol.

        Returns:
        -------
        np.ndarray
            Array of complex constellation points.
        """
        N = int(np.sqrt(self.M))
        if N % 2 != 0:
            raise ValueError("The square root of M must be an even number.")

        levels = np.arange(-N + 1, N, 2)
        constellation = []
        for i in levels:
            for q in levels:
                constellation.append(complex(i, q))

        constellation = np.array(constellation)
        avg_power = np.mean(np.abs(constellation) ** 2)
        return constellation / np.sqrt(avg_power)
    
    def generate_symbol_mapping(self):
        """
        Maps all bit combinations to the corresponding constellation symbols.

        Returns:
        -------
        dict
            A dictionary mapping bit strings to constellation points.
        """
        bit_combinations = [np.binary_repr(i, width=self.bits_per_symbol) for i in range(self.M)]
        return {bits: self.constellation[i] for i, bits in enumerate(bit_combinations)}

    def parameter(self, data: np.ndarray, fc: int = 584e6, gps: np.ndarray = None) -> dict:
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

        bits = []
        for symbol in self.frequencies:
            distances = np.abs(self.constellation - symbol)
            index = np.argmin(distances)
            bits_str = np.binary_repr(index, width=self.bits_per_symbol)
            bits.extend([int(b) for b in bits_str])

        bits = np.array(bits)

        num_symbols = len(bits) // self.bits_per_symbol
        symbols = []
        for i in range(num_symbols):
            bit_group = ''.join(map(str, bits[i * self.bits_per_symbol:(i + 1) * self.bits_per_symbol]))
            symbol = self.symbol_mapping[bit_group]
            symbols.append(symbol)

        transmitted_bits = np.random.randint(0, 2, len(data) * self.bits_per_symbol)
        transmitted_symbols = np.array(symbols)

        #---------MER calculation-------#

        error_vector = bits - transmitted_symbols
        signal_power = np.mean(np.abs(transmitted_symbols) ** 2)
        noise_power = np.mean(np.abs(error_vector) ** 2)
        mer = 10 * np.log10(signal_power / noise_power)

        #---------BER calculation---------#

        errors = np.sum(transmitted_bits != bits)
        total_bits = len(transmitted_bits)
        ber = errors / total_bits
        
        self.parameters = {
                'time': [],
                'freq': [],
                'power': [],
                'c/n': [],
                'mer': [],
                'ber': []
        }

        t0 = time.strftime('%X')

        f, Pxx = self.pros.welch(data, self.sample_rate)
        f = (f + fc) / 1e6

        # plt.semilogy(f, Pxx)
        

        # plt.scatter(peak_freqs, peak_powers, c='green')
        # plt.axhline(threshold, c='red')

        for center_freq in self.frequencies:
            index = np.where(np.isclose(f, center_freq, atol=0.01))[0][0]

            lower_index = np.argmin(np.abs(f - (f[index] - 4)))
            upper_index = np.argmin(np.abs(f - (f[index] + 4)))

            freq_range = f[lower_index:upper_index + 1]
            Pxx_range = Pxx[lower_index:upper_index + 1]     

            # plt.semilogy(freq_range, Pxx_range) 

            power = np.trapz(Pxx_range, freq_range)

            self.parameters['time'].append(t0)
            self.parameters['freq'].append(center_freq)
            self.parameters['c/n'].append(10 * np.log10(power / Pxx[0]))
            self.parameters['power'].append(10 * np.log10(power))
            self.parameters['mer'].append(mer)
            self.parameters['ber'].append(ber)

        

        # plt.show()

        return f, Pxx, self.parameters