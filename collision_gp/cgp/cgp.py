import torch
import gpytorch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from .use_cases import GPModel, PGLikelihood
from ..entities import CGPInfo


class CGP:
    def __init__(self, m: int, dof: int) -> None:
        self.model: GPModel = GPModel(torch.Tensor(np.random.rand(m, dof))*2.0-1, dof)
        self.likelihood: PGLikelihood = PGLikelihood()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def train(self, input_data: np.ndarray, labels: np.ndarray) -> None:
        if not np.all(np.logical_and(input_data >= -1, input_data <= 1)):
            raise ValueError('input data must be in the interval [-1, 1]')

        if not np.all(np.logical_or(labels == 0, labels == 1)):
            raise ValueError('labels must be either 0 or 1')

        # Tensor initialization
        train_x: torch.Tensor = torch.Tensor(input_data)
        train_y: torch.Tensor = torch.Tensor(labels)

        if torch.cuda.is_available():
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        # Training batch
        training_batch: int = 256
        dataset = TensorDataset(train_x, train_y)
        data_loader = DataLoader(dataset, batch_size=training_batch, shuffle=True)

        #
        variational_ngd_optimizer = gpytorch.optim.NGD(self.model.variational_parameters(), num_data=train_y.size(0), lr=0.5)
        hyperparameter_optimizer = torch.optim.Adam([{'params': self.model.hyperparameters(), 'params': self.likelihood.parameters()},], lr=0.1)

        # Set into training mode
        self.model.train()
        self.likelihood.train()

        loss_function = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        epochs: int = 10
        for i in range(epochs):
            losses = []

            for x_batch, y_batch in data_loader:
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -loss_function(output, y_batch)  # type: ignore
                losses.append(loss.item())
                loss.backward()

                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

            print(f'iteration {i+1}/{epochs} ({i/epochs*100:.2f}%. Error: {np.mean(losses)})')

    def predict(self, input_data: np.ndarray, beta: float = 0.5) -> CGPInfo:
        # TODO: check length of the input data
        # Initialize tensors
        test_x: torch.Tensor = torch.Tensor(input_data)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            predictions = self.likelihood(self.model(test_x))
            means = predictions.mean.cpu().numpy()  # type: ignore
            deviation = predictions.stddev.cpu().numpy()  # type: ignore
            variance = predictions.variance.cpu().numpy()  # type: ignore

        decision: np.ndarray = means + beta*deviation

        return CGPInfo(decision=decision,
                       mean=means,
                       deviation=deviation,
                       variance=variance,
                       )

    def save(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename+'_model.pth')
        torch.save(self.likelihood.state_dict(), filename+'_likelihood.pth')

    def load(self) -> None:
        pass
