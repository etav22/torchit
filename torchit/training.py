"""
Functions to facilitiate training PyTorch model
"""
from timeit import default_timer as timer
from typing import Literal, Tuple

import torch
from torch import nn
from tqdm import tqdm


class TrainEval:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metric_fx: Literal["Accuracy", "MAE", "MSE"] = None,
        device: str = "cpu",
    ) -> None:
        """Conducts the training procedure for a PyTorch model and also
        adds some evaluation functionality.

        Args:
            model (torch.nn.Module): PyTorch model
            train_dataloader (torch.utils.data.DataLoader):
                Iterable dataset containing the train data.
            test_dataloader (torch.utils.data.DataLoader):
                Iterable dataset containing the test data.
            criterion (torch.nn.Module): Also known as the loss function
                - calculates the difference between predicted and true.
            metric (Literal[&quot;Accuracy&quot;,
                    &quot;MAE&quot;,
                    &quot;MSE&quot;]):
                Sets metric to be used during training. Defaults to None.
            device (str, optional): Device to train on. Defaults to "cpu".
        """

        # Base objects for training
        self.model = model.to(
            device
        )  # if model not already on device, which defaults to cpu
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.device = device
        # TODO Instantiate metric_fx

        # Instantiate loss and metric dictionaries:
        self.loss = {"train": [], "test": []}
        self.metric = {"train": [], "test": []}

        # Instantiate run time
        self.run_time = 0

    def _train_epoch(self, optimizer: nn.Module) -> Tuple[float, float]:
        """Trains the model for a single epoch.

        Args:
            optimizer (nn.Module): Optimizes the training loss of the model.

        Returns:
            Tuple[float, float]: Training loss and training metric.
        """
        # Set the model to train mode
        self.model.train()

        # Instantiate loss and metric:
        train_loss, train_metric = 0, 0

        # Iterate over the dataloader
        for X_train, y_train in self.train_dataloader:
            # Send the data to the device
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            # Forward pass
            y_train_pred = self.model(X_train)

            # Calculate the loss
            loss = self.criterion(y_train_pred, y_train)
            train_loss += loss

            # TODO: Calculate the train metric

            # Zero optimizer, backward pass, and step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate the averge loss and metric of the epoch:
        train_loss /= len(self.train_dataloader)
        train_metric /= len(self.train_dataloader)

        return train_loss, train_metric

    def _test_epoch(self) -> Tuple[float, float]:
        """Tests the model for a single epoch

        Returns:
            Tuple[float, float]: Test loss and test metric.
        """

        # Set the model to eval mode
        self.model.eval()

        # Instantiate the test loss and test metric
        test_loss, test_metric = 0, 0

        # Set inference mode
        with torch.inference_mode():
            # Iterate over the test loader
            for X_test, y_test in self.test_dataloader:
                # Send the tensors to the device
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)

                # Forward pass
                y_pred_test = self.model(X_test)

                # Calculate the test loss
                loss = self.criterion(y_pred_test, y_test)
                test_loss += loss

                # TODO: Calculate the test metric

            # Calculate the average loss and metric of the epoch:
            test_loss /= len(self.test_dataloader)
            test_metric /= len(self.test_dataloader)

        return test_loss, test_metric

    def _time_model(self, start_time: float, end_time: float) -> float:
        """Calculates the total training time of the model

        Args:
            start_time (float): Time model starts training
            end_time (float): Time model stops training

        Returns:
            float: Total training time in seconds
        """
        # Calculate the total run time
        total_run_time = start_time - end_time

        # Print out the total time
        print(f"Time to run model: {total_run_time:.5f} seconds")

        return total_run_time

    def train(self, num_epochs: int, optimizer: torch.nn.Module) -> None:
        # Start timer
        start = timer()

        for epoch in tqdm(range(num_epochs)):
            # Print out epoch:
            print(f"Epoch: {epoch}\n-------------------------")

            # Train Step
            train_loss, train_metric = self._train_epoch(optimizer=optimizer)
            self.loss["train"].append(train_loss)
            self.loss["test"].append(train_metric)

            # Test Step
            test_loss, test_metric = self._test_epoch()
            self.metric["train"].append(test_loss)
            self.metric["test"].append(test_metric)

            # Print out the train and test losses+metrics:
            print(f"Train Loss: {train_loss} | Train Metric: {train_metric}")
            print(f"Test Loss: {test_loss} | Test Metric: {test_metric}")

        # End timer
        end = timer()

        # Gather final time
        self.run_time = self._time_model(start_time=start, end_time=end)

    def evaluate():
        pass
