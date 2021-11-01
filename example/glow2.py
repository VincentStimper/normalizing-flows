import numpy as np
import torch
import torchvision
from tqdm import tqdm

import normflow as nf

# Construct Model
# Define flows
L = 3
K = 16
torch.manual_seed(0)

input_shape = (3, 32, 32)
channels = 3
hidden_channels = 256
split_mode = 'channel'
scale = True
num_classes = 10
# Set up flows, distributions and merge operations
q0 = []
merges = []
flows = []
for i in range(L):
    flows_ = []
    for j in range(K):
        flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                      split_mode=split_mode, scale=scale)]
    flows_ += [nf.flows.Squeeze()]
    flows += [flows_]
    latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                    input_shape[2] // 2 ** (L - i))
    if i > 0:
        merges += [nf.flows.Merge()]
        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                        input_shape[2] // 2 ** (L - i))
    else:
        latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                        input_shape[2] // 2 ** L)
    q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

# Construct flow model
model = nf.MultiscaleFlow(q0, flows, merges)
# End Construct Model
logit = nf.utils.Logit(alpha=0.05)
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
device = torch.device('cpu')
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), nf.utils.Jitter(), logit, nf.utils.ToDevice(device)])
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
train_iter = iter(trainloader)

# Train model
max_iter = 2000

loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for i in tqdm(range(max_iter)):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(trainloader)
        x, y = next(train_iter)
    print(x.shape)
    print(type(x))
    print(y.shape)
    print(type(y))
