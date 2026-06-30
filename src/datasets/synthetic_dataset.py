import torch
from torch.utils.data import Dataset



class RFDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        window=256
    ):

        self.X=X
        self.y=y
        self.window=window



    def __len__(self):

        return len(self.y)



    def __getitem__(self,index):

        signal=self.X[index]


        signal=signal[:,:self.window]


        # normalize

        signal = (
            signal -
            signal.mean()
        ) / (
            signal.std()+1e-8
        )


        return {

            "x":signal,

            "y":self.y[index],

            "metadata":
            {
                "synthetic":True,
                "device":int(self.y[index])
            }

        }