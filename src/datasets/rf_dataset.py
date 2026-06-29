import torch
from torch.utils.data import Dataset
import numpy as np
import glob



class RFDataset(Dataset):

    def __init__(self, path, window_size=256):

        self.files = glob.glob(
            path + "/*.npy"
        )

        self.window_size = window_size


        self.samples=[]


        for label,file in enumerate(self.files):

            data=np.load(file)


            for i in range(
                0,
                len(data)-window_size,
                window_size
            ):

                window=data[i:i+window_size]


                self.samples.append(
                    (
                        window,
                        label
                    )
                )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self,index):

        x,y=self.samples[index]


        # complex -> I/Q channels

        x=np.stack(
            [
                x.real,
                x.imag
            ],
            axis=0
        )


        x=torch.tensor(
            x,
            dtype=torch.float32
        )


        return {
            "x":x,
            "y":torch.tensor(y),
            "metadata":{
                "synthetic":True
            }
        }