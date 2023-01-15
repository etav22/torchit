"""
Functions to facilitiate training PyTorch model
"""
import warnings
# General imports
from timeit import default_timer as timer
from typing import Dict, Literal, Tuple

# Torch imports
import torch
import torchinfo
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
        self.trained = False
        # TODO Instantiate metric_fx

        # Instantiate loss and metric dictionaries:
        self.loss = {"train": [], "test": []}
        self.metric = {"train": [], "test": []}

        # Instantiate run time
        self.run_time = 0

    def _train_epoch(self, optimizer: nn.Module) -> Tuple[float, float]:
        """Trains the model for a single epoch. In training, both
        loss and a pre-defined test-metric are calculated in each batch.
        The loss and test-metrics are then averaged over the length of the dataloader
        at the end of the epoch.

        Args:
            optimizer (nn.Module): Optimizes the training loss of the model.

        Returns:
            Tuple[float, float]: Average  training loss and training metric.
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
        """Tests the model for a single epoch. In testing, both
        loss and a pre-defined test-metric are calculated in each batch.
        The loss and test-metrics are then averaged over the length of the dataloader
        at the end of the epoch.

        Returns:
            Tuple[float, float]: Average test loss and test metric.
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
        """Calculates the total training time of the model.

        Args:
            start_time (float): Time model starts training.
            end_time (float): Time model stops training.

        Returns:
            float: Total training time in seconds.
        """
        # Calculate the total run time
        total_run_time = end_time - start_time

        # Print out the total time
        print(f"Time to run model: {total_run_time:.5f} seconds")

        return total_run_time

    def train(self, num_epochs: int, optimizer: torch.nn.Module) -> None:
        """_summary_

        Args:
            num_epochs (int): _description_
            optimizer (torch.nn.Module): _description_
        """
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

        # Switch model to trained
        self.trained = True

        # Gather final time
        self.run_time = self._time_model(start_time=start, end_time=end)

    def predict(self) -> Dict[str, list]:
        """Generates a list of predictions based on the test dataloader that is
        passed into the model during initialization. This function then returns
        a dictionary containing two keys:

            - predictions
            - labels

        Each contain a list for the predicted and true value for each item in the
        test dataloader

        Returns:
            Dict[str, list]: Results dictionary with both predictions and labels.
        """
        # Output a warning for non-trained model
        if self.trained is False:
            warnings.warn(
                "The model is not yet trained. Evaluation will be as good as random"
            )

        # Instantiate dictionary to house predictions
        results = {"predictions": [], "labels": []}

        # Set the model to eval and use inference
        self.model.eval()
        with torch.inference_mode():
            # Itereate over the dataloader
            for X_eval, y_eval in tqdm(self.test_dataloader):
                # Send the data to the device
                X_eval, y_eval = X_eval.to(self.device), y_eval.to(self.device)

                # Make the predictions
                y_preds_eval = self.model(X_eval)

                # Convert the models to predictions:
                y_labels = torch.argmax(y_preds_eval, dim=1)

                # Append both the predictions an output list
                results["predictions"].append(y_labels.cpu())
                results["labels"].append(y_eval)
        
        return results

    def info(self, input_size: tuple) -> torchinfo.ModelStatistics:
        """Provdies an information summary for the torch model by using
        the torchinfo.summary function. The info returned is:

            - Model Layers
            - Number of params
            - Model size

        Args:
            input_size (tuple): Input size of the image tensors.

        Returns:
            torchinfo.ModelStatistics: Informative summary of the model.
        """
        return torchinfo.summary(self.model, input_size=input_size)
