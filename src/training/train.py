import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.toy_generator import create_dataset
from src.datasets.synthetic_dataset import RFDataset
from src.models.cnn1d import RF_CNN
from src.evaluation.metrics import calculate_metrics



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
        "Epoch", epoch,
        "Loss:", total
    )


model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for batch in loader:

        x=batch["x"]

        y=batch["y"]

        pred=model(x)

        labels=pred.argmax(dim=1)

        y_true.extend(y.tolist())

        y_pred.extend(labels.tolist())


metrics=calculate_metrics(
    y_true,
    y_pred
)

out_dir=Path("results/week1_toy_cnn")

out_dir.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    out_dir / "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


print(
    "Saved metrics to",
    out_dir / "metrics.json"
)
