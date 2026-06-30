from src.toy_generator import create_dataset
from src.datasets.synthetic_dataset import RFDataset


X,y=create_dataset()

dataset=RFDataset(X,y)

print(dataset[0])