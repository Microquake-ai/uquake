import numpy as np


class BrunePulse:

    def __init__(self, f_c: float = 1, time: np.ndarray = np.arange(0, 10, 0.01)):
        pass


def generate_brune_pulse(f_c, time):
    """
    Generates the Brune pulse for a given corner frequency and time array.

    Parameters:
    f_c (float): Corner frequency of the event.
    time (numpy array): Array of time values at which the pulse is evaluated.

    Returns:
    numpy array: The Brune pulse evaluated at the given time points.
    """
    # Calculate the decay constant
    tau = 1 / (2 * np.pi * f_c)

    # Generate the Brune pulse
    pulse = time * np.exp(-time / tau)
    return pulse