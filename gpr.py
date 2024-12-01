import math
import matplotlib.pyplot as plt
import gpytorch.models
import numpy as np
import torch
from gpytorch.kernels import RBFKernel
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
import glob

# 使用glob自动获取匹配的文件路径
input_files = glob.glob("data/Input-Data/NACA0015-AoA15.00-Ma0.378-Re1.95d6-*-input.plt")
output_files = glob.glob("data/Output-Data/NACA0015-AoA15.00-Ma0.378-Re1.95d6-*-output.plt")

print(input_files)
print(output_files)

# 读取所有输入文件和输出文件
input_data = []
output_data = []

for input_file in input_files:
    with open(input_file, "r") as file:
        input_lines = file.readlines()

    for line in input_lines:
        line = line.strip()
        if not line.startswith("VARIABLES") and not line.startswith("ZONE"):
            try:
                input_data.append([float(x) for x in line.split()])
            except ValueError:
                continue  # 跳过无法转换的行

for output_file in output_files:
    with open(output_file, "r") as file:
        output_lines = file.readlines()

    for line in output_lines:
        line = line.strip()
        if not line.startswith("VARIABLES") and not line.startswith("ZONE"):
            try:
                output_data.append([float(x) for x in line.split()])
            except ValueError:
                continue  # 跳过无法转换的行

X_all = np.array(input_data)  # (num_samples, 44)
y_all = np.array(output_data)  # (num_samples, 3)

x_mean = X_all.mean(0)
x_std = X_all.std(0)

y_mean = y_all.mean(0)[2]
y_std = y_all.std(0)[2]


X_train = torch.tensor(X_all[:5867, :], dtype=torch.double)
y_train = torch.tensor(y_all[:5867, 2], dtype=torch.double)
X_train_norm = (X_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std

X_test = torch.tensor(X_all[5867:, :], dtype=torch.double)
y_test = torch.tensor(y_all[5867:, 2], dtype=torch.double)
X_test_norm = (X_test - x_mean) / x_std
y_test_norm = (y_test - y_mean) / y_std

print("train input Shape:", X_train.shape)
print("train output Shape:", y_train.shape)

print("test input Shape:", X_test.shape)
print("test output Shape:", y_test.shape)


# GPR模型定义
class GPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean, kernel, likelihood):
        super(GPR, self).__init__(train_x, train_y, likelihood)
        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.likelihood(self(x))


def train(model, X_train, y_train, n_iter=500, lr=0.1):
    model.train()
    likelihood.train()
    training_parameters = model.parameters()
    optimizer = torch.optim.Adam(training_parameters, lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Iter %d/%d   Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_iter, loss.item(),
                model.kernel.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()

            ))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean_fn = gpytorch.means.ConstantMean()
kernel_fn = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
likelihood = gpytorch.likelihoods.GaussianLikelihood()


model = GPR(X_train_norm, y_train, mean_fn, kernel_fn, likelihood).to(device)


X_train_norm = X_train_norm.to(device)
y_train_norm = y_train.to(device)
X_test_norm = X_test_norm.to(device)
y_test_norm = y_test.to(device)


train(model, X_train_norm, y_train_norm)

# 测试集
model.eval()
likelihood.eval()
with torch.no_grad():

    y_pred = model.predict(X_test_norm)
    n_pred = min(100, len(y_pred.mean))
    n_train = len(y_train)

    lower, upper = y_pred.confidence_region()

    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax.plot(range(n_train), y_train.numpy(), 'k*', label='Training Data')
    ax.plot(np.arange(n_pred), y_pred.mean.cpu().numpy()[:n_pred], 'b', label='predict Data')
    # ax.scatter(np.arange(n_pred),
    #            y_test.cpu().numpy()[:n_pred],
    #            color='k',
    #            s=4,
    #            label='Test Data')
    ax.plot(np.arange(n_pred), y_test.cpu().numpy()[:n_pred], 'k', label='Test Data')

    ax.fill_between(np.arange(n_pred), lower.cpu().numpy()[:n_pred], upper.cpu().numpy()[:n_pred], color='gray', alpha=0.5,
                    label='95% Confidence Interval')
    ax.legend()
    plt.show()
