import torch
import random
import matplotlib.pyplot as plt
from typing import Tuple

def visualize_samples(
    dataset: torch.utils.data.Dataset,
    num_samples: int,
    figsize: Tuple[int, int]=(10, 6),
    seed: int=42
):
    """Returns a subplot of randomly sampled images from a torch dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset from which images are sampled.
        num_samples (int): Number of images to display in subplots.
        figsize (Tuple[int, int], optional): Size of subplot figure. Defaults to (10, 6).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Raises:
        ValueError: If more than 10 samples are given, the function does not plot. 
    """

    # Check the number of samples:
    if num_samples > 9:
        raise ValueError("num_samples: Sample size must be less than 9")

    # First select the random samples
    random.seed(seed)
    random_idx = random.sample(range(len(dataset)), k=num_samples)

    # We have to split the size of the graph with some sense:
    if num_samples % 3 == 0:
        n_row = int(num_samples / 3)
        n_col = 3
    elif num_samples == 2:  # have this "exception" because two graphs look better side-by-side
        n_row = 1
        n_col = 2
    elif num_samples % 2 == 0 : 
        n_row = 2
        n_col = int(num_samples / 2)

    # Generate the subplot object accordingly
    fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
    ax = ax.ravel()

    # Iterate over the list of random indeces and create a graph for each
    for i, idx in enumerate(random_idx):
        
        # Gather the image and label:
        img, label = dataset[idx]

        # Plot the image
        ax[i].set_title(dataset.classes[label])
        ax[i].imshow(img.squeeze(), cmap="gray")

    # Tighten the axes and show the plot
    plt.tight_layout()
    plt.show()






