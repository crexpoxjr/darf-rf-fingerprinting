import numpy as np
import torch


def generate_device_signal(
    device_id,
    samples=4096
):

    t = np.arange(samples)


    # Device fingerprint parameters
    cfo = device_id * 0.001
    phase = device_id * 0.2
    amplitude = 1 + device_id * 0.05


    iq = (
        amplitude *
        np.exp(
            1j *
            (
                2*np.pi*cfo*t
                + phase
            )
        )
    )


    noise = (
        np.random.randn(samples)
        +
        1j*np.random.randn(samples)
    ) * 0.05


    iq += noise


    # Convert complex -> I/Q channels

    x = np.stack(
        [
            iq.real,
            iq.imag
        ],
        axis=0
    )


    return torch.tensor(
        x,
        dtype=torch.float32
    )



def create_dataset(
    devices=5,
    samples_per_device=100
):

    X=[]
    y=[]


    for device in range(devices):

        for _ in range(samples_per_device):

            signal = generate_device_signal(device)

            X.append(signal)
            y.append(device)


    return torch.stack(X), torch.tensor(y)



if __name__=="__main__":

    X,y=create_dataset()

    print(X.shape)
    print(y.shape)
