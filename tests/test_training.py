from torchit.training import TrainEval
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as T
import torch

# Instantiate a small model
class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_size, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


def test_run():
    # Test with MNIST Dataset
    DATA_ROOT = "test_data"
    BATCH_SIZE = 32
    NUM_EPOCHS = 3

    # Set a device
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Set the datasets
    train_dataset = MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=T.ToTensor()
    )
    test_dataset = MNIST(
        root=DATA_ROOT, 
        train=False, 
        download=True, 
        transform=T.ToTensor()
    )

    # Set the dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Instantiate the small model
    model = NeuralNet(input_size=28*28, hidden_units=5, output_shape=len(test_dataset.classes))

    # Set a criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    # Instantiate the traineval class
    model_train_eval = TrainEval(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
    )

    # Train the model
    model_train_eval.train(num_epochs=NUM_EPOCHS, optimizer=optimizer)

    # Check results
    print(model_train_eval.run_time)


