import torchvision.transforms as T
from torchvision.datasets import MNIST

from torchit.visualizations import visualize_samples


def test_run():
    # Test with MNIST Dataset
    DATA_ROOT = "test_data"

    # Set the datasets
    dataset = MNIST(root=DATA_ROOT, train=True, download=True, transform=T.ToTensor())

    # Visualize a set of samples
    visualize_samples(
        dataset=dataset,
        num_samples=9,
    )
