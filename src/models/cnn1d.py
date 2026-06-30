import torch.nn as nn


class RF_CNN(nn.Module):

    def __init__(
        self,
        classes=5
    ):

        super().__init__()


        self.net=nn.Sequential(

            nn.Conv1d(
                2,
                16,
                5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),


            nn.Conv1d(
                16,
                32,
                5,
                padding=2
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )


        self.fc=nn.Linear(
            32,
            classes
        )


    def forward(self,x):

        x=self.net(x)

        x=x.squeeze(-1)

        return self.fc(x)