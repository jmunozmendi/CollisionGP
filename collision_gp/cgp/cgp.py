import torch
import numpy as np

from torch import Tensor
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.optim import NGD
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from ..core import Logger
from .use_cases import GPModel, PGLikelihood
from ..entities import CGPInfo


class CGP:
    def __init__(self, m: int, dof: int) -> None:
        self.model: GPModel = GPModel(Tensor(np.random.uniform(-1, 1, (m, dof))), dof)
        self.likelihood: PGLikelihood = PGLikelihood()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def train(self,
              input_data: np.ndarray,
              labels: np.ndarray,
              epochs: int = 10,
              batch_size: int = 256,
              variational_lr: float = 0.5,
              hyperparameter_lr: float = 0.1,
              ) -> None:

        # Function requirements
        if not np.all(np.logical_and(input_data >= -1, input_data <= 1)):
            raise ValueError('input data must be in the interval [-1, 1]')

        if not np.all(np.logical_or(labels == 0, labels == 1)):
            raise ValueError('labels must be either 0 or 1')

        # Tensor initialization
        train_x: Tensor = Tensor(input_data).cuda() if torch.cuda.is_available() else Tensor(input_data)
        train_y: Tensor = Tensor(labels).cuda() if torch.cuda.is_available() else Tensor(labels)

        # Set into training mode
        self.model.train()
        self.likelihood.train()

        # Optimizers
        variational_ngd_optimizer: NGD = NGD(self.model.variational_parameters(), num_data=train_y.size(0), lr=variational_lr)
        hyperparameter_optimizer: Adam = Adam([{'params': self.model.hyperparameters(), 'params': self.likelihood.parameters()},], lr=hyperparameter_lr)
        loss_function: VariationalELBO = VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        # Training batch
        dataset: TensorDataset = TensorDataset(train_x, train_y)
        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i in range(epochs):
            for x_batch, y_batch in data_loader:
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()

                output: MultivariateNormal = self.model(x_batch)

                loss: Tensor = -loss_function(output, y_batch)  # type: ignore
                loss.backward()

                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

            Logger.info(f'Training in progress... {int((i/epochs)*100)}%', flush=True)

    def predict(self, input_data: np.ndarray, beta: float = 0.5) -> CGPInfo:
        # TODO: check length of the input data
        # Initialize tensors
        test_x: torch.Tensor = torch.Tensor(input_data)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean.cpu().numpy()  # type: ignore
            deviation = predictions.stddev.cpu().numpy()  # type: ignore
            variance = predictions.variance.cpu().numpy()  # type: ignore

        decision: np.ndarray = mean + beta*deviation

        return CGPInfo(decision=decision,
                       mean=mean,
                       deviation=deviation,
                       variance=variance,
                       )

    def save(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename+'_model.pth')
        torch.save(self.likelihood.state_dict(), filename+'_likelihood.pth')

    def load(self) -> None:
        pass
