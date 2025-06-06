import cudaq
import numpy as np

class CustomNoiseChannel(cudaq.KrausChannel):
    num_parameters = 1
    num_targets = 1
    
    def __init__(self, params: list[float]):
        cudaq.KrausChannel.__init__(self)
        # Example: Create Kraus ops based on params
        p = params[0]
        k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                      dtype=np.complex128)
        k1 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]],
                      dtype=np.complex128)

        # Create KrausOperators and add to channel
        self.append(cudaq.KrausOperator(k0))
        self.append(cudaq.KrausOperator(k1))

        self.noise_type = cudaq.NoiseModelType.Unknown

def main():
    noise = cudaq.NoiseModel()
    noise.register_channel(CustomNoiseChannel)
    
if __name__ == "__main__":
    main()