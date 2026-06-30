import torch
from torch.utils.data import DataLoader

from src.toy_generator import create_dataset
from src.datasets.synthetic_dataset import RFDataset
from src.models.cnn1d import RF_CNN



X,y=create_dataset()

dataset=RFDataset(X,y)

loader=DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)



model=RF_CNN()


optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


loss_fn=torch.nn.CrossEntropyLoss()



for epoch in range(10):

    total=0

    for batch in loader:

        x=batch["x"]

        y=batch["y"]


        pred=model(x)

        loss=loss_fn(
            pred,
            y
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total+=loss.item()


    print(
        epoch,
        total
    )