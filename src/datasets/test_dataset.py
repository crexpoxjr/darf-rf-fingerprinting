from src.datasets.rf_dataset import RFDataset


dataset=RFDataset(
    "data/src/toy-data"
)


sample=dataset[0]


print(sample["x"].shape)
print(sample["y"])
print(sample["metadata"])