import numpy as np
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = "data/toy-data"


def generate_signal(
    num_samples=10000,
    frequency_offset=0.0,
    phase_offset=0.0,
    amplitude=1.0,
    iq_imbalance=0.0,
    snr_db=30
):
    """
    Generate synthetic I/Q samples
    """

    t = np.arange(num_samples)

    # Base carrier
    signal = np.exp(
        1j * 2 * np.pi * frequency_offset * t
    )


    # Phase fingerprint
    signal *= np.exp(
        1j * phase_offset
    )


    # Amplitude fingerprint
    signal *= amplitude


    # I/Q imbalance
    i = signal.real
    q = signal.imag * (1 + iq_imbalance)

    signal = i + 1j*q


    # Add noise
    power = np.mean(np.abs(signal)**2)

    noise_power = power / (
        10**(snr_db/10)
    )

    noise = np.sqrt(noise_power/2) * (
        np.random.randn(num_samples)
        +
        1j*np.random.randn(num_samples)
    )

    return signal + noise



def create_devices():

    devices = [
        {
            "frequency_offset":0.0001,
            "phase_offset":0.0,
            "amplitude":1.0,
            "iq_imbalance":0.00
        },

        {
            "frequency_offset":0.0003,
            "phase_offset":0.2,
            "amplitude":0.9,
            "iq_imbalance":0.05
        },

        {
            "frequency_offset":-0.0002,
            "phase_offset":-0.15,
            "amplitude":1.1,
            "iq_imbalance":-0.04
        },

        {
            "frequency_offset":0.0005,
            "phase_offset":0.1,
            "amplitude":0.95,
            "iq_imbalance":0.08
        },

        {
            "frequency_offset":-0.0004,
            "phase_offset":-0.25,
            "amplitude":1.05,
            "iq_imbalance":-0.06
        }
    ]


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    for idx, params in enumerate(devices):

        samples = generate_signal(**params)

        np.save(
            f"{OUTPUT_DIR}/device_{idx}.npy",
            samples
        )

        print(
            f"Saved device {idx}"
        )


if __name__ == "__main__":
    create_devices()


for i in range(5):

    x=np.load(
        f"data/toy-data/device_{i}.npy"
    )

    plt.figure(figsize=(10,3))

    plt.plot(
        x.real[:500],
        label="I"
    )

    plt.plot(
        x.imag[:500],
        label="Q"
    )

    plt.title(
        f"Device {i}"
    )

    plt.legend()

    plt.savefig(
        f"device_{i}.png"
    )
