import gpytorch
import torch


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, dof: int) -> None:
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module: gpytorch.means.ZeroMean = gpytorch.means.ZeroMean()
        self.covar_module: gpytorch.kernels.ScaleKernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=dof))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # type: ignore
