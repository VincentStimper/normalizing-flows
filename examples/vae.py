from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from normflows.flows import Planar, Radial, MaskedAffineFlow, BatchNorm
import argparse
from datetime import datetime
import os
from normflows import nets
import pandas as pd
import random

parser = argparse.ArgumentParser(description="FlowVAE implementation on MNIST")
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    metavar="N",
    help="Training batch size (default: 256)",
)
parser.add_argument(
    "--latent-size",
    type=int,
    default=40,
    metavar="N",
    help="Latent dimension size (default: 40)",
)
parser.add_argument(
    "--K", type=int, default=10, metavar="N", help="Number of flows (default: 10)"
)
parser.add_argument(
    "--flow",
    type=str,
    default="Planar",
    metavar="N",
    help="Type of flow (default: Planar)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=15,
    metavar="N",
    help="Nr of training epochs (default: 15)",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    metavar="N",
    help="Dataset to train and test on (mnist, cifar10 or cifar100) (default: mnist)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=15, metavar="S", help="Random Seed (default: 1)"
)
parser.add_argument(
    "--log-intv",
    type=int,
    default=20,
    metavar="N",
    help="Training log status interval (default: 20",
)
parser.add_argument(
    "--experiment_mode",
    type=bool,
    default=False,
    metavar="N",
    help="Experiment mode (conducts 10 runs and saves results as DataFrame (default: False)",
)
parser.add_argument(
    "--runs",
    type=int,
    default=10,
    metavar="N",
    help="Number of runs in experiment_mode (experiment_mode has to be turned to True to use) (default: 10)",
)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


class SimpleFlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        ld = 0.0
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_

        return z, ld


class BinaryTransform:
    def __init__(self, thresh=0.5):
        self.thresh = thresh

    def __call__(self, x):
        return (x > self.thresh).type(x.type())


class ColourNormalize:
    def __init__(self, a=0.0, b=0.0):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.b - self.a) * x / 255 + self.a


if args.dataset == "mnist":
    img_dim = 28
    dtf = transforms.Compose([transforms.ToTensor(), BinaryTransform()])
elif args.dataset == "cifar10" or args.dataset == "cifar100":
    img_dim = 8
    dtf = transforms.Compose(
        [
            transforms.RandomCrop([8, 8]),
            transforms.ToTensor(),
            ColourNormalize(0.0001, 1 - 0.0001),
        ]
    )
else:
    raise ValueError("The only dataset calls supported are: mnist, cifar10, cifar100")


def extract_cifar_patch(tensor, target_size):
    x = random.randint(0, 32 - target_size)
    y = random.randint(0, 32 - target_size)
    return tensor[x : x + target_size, y : y + target_size, :]


# Training
def flow_vae_datasets(
    id,
    download=True,
    batch_size=args.batch_size,
    shuffle=True,
    transform=dtf,
    patch_size=None,
):
    data_d_train = {
        "mnist": datasets.MNIST(
            "datasets", train=True, download=True, transform=transform
        ),
        "cifar10": datasets.CIFAR10(
            "datasets", train=True, download=True, transform=transform
        ),
        "cifar100": datasets.CIFAR100(
            "datasets", train=True, download=True, transform=transform
        ),
    }
    data_d_test = {
        "mnist": datasets.MNIST(
            "datasets", train=False, download=True, transform=transform
        ),
        "cifar10": datasets.CIFAR10(
            "datasets", train=False, download=True, transform=transform
        ),
        "cifar100": datasets.CIFAR100(
            "datasets", train=False, download=True, transform=transform
        ),
    }

    # training_data = data_d_train.get(id)
    # test_data = data_d_test.get(id)
    # if patch_size is not None:
    # training_data.data = np.stack(
    # [extract_cifar_patch(training_data.data[i, :, :], patch_size) for i in range(len(training_data.data))])
    # test_data.data = np.stack(
    # [extract_cifar_patch(test_data.data[i, :, :], patch_size) for i in range(len(test_data.data))])

    train_loader = torch.utils.data.DataLoader(
        data_d_train.get(id), batch_size=batch_size, shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        data_d_test.get(id), batch_size=batch_size, shuffle=shuffle
    )
    return train_loader, test_loader


class FlowVAE(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(img_dim**2, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
        )
        self.f1 = nn.Linear(256, args.latent_size)
        self.f2 = nn.Linear(256, args.latent_size)
        self.decode = nn.Sequential(
            nn.Linear(args.latent_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim**2),
        )
        self.flows = flows

    def forward(self, x):
        # Encode
        mu, log_var = self.f1(
            self.encode(x.view(x.size(0) * x.size(1), img_dim**2))
        ), self.f2(self.encode(x.view(x.size(0) * x.size(1), img_dim**2)))

        # Reparameterize variables
        std = torch.exp(0.5 * log_var)
        norm_scale = torch.randn_like(std)
        z_0 = mu + norm_scale * std

        # Flow transforms
        z_, log_det = self.flows(z_0)
        z_ = z_.squeeze()

        # Q0 and prior
        q0 = Normal(mu, torch.exp((0.5 * log_var)))
        p = Normal(0.0, 1.0)

        # KLD including logdet term
        kld = (
            -torch.sum(p.log_prob(z_), -1)
            + torch.sum(q0.log_prob(z_0), -1)
            - log_det.view(-1)
        )
        self.test_params = [
            torch.mean(-torch.sum(p.log_prob(z_), -1)),
            torch.mean(torch.sum(q0.log_prob(z_0), -1)),
            torch.mean(log_det.view(-1)),
            torch.mean(kld),
        ]

        # Decode
        z_ = z_.view(z_.size(0), args.latent_size)
        zD = self.decode(z_)
        out = torch.sigmoid(zD)

        return out, kld


def logit(x):
    return torch.log(x / (1 - x))


def bound(rce, x, kld, beta):
    if args.dataset == "mnist":
        return (
            F.binary_cross_entropy(rce, x.view(-1, img_dim**2), reduction="sum")
            + beta * kld
        )
    elif args.dataset == "cifar10" or args.dataset == "cifar100":
        # return (- torch.distributions.Normal(x.view(-1, img_dim ** 2), 1.).log_prob(rce)).sum() + beta * kld
        return F.mse_loss(rce, x, reduction="sum") + beta * kld


if args.flow == "Planar":
    flows = SimpleFlowModel([Planar((args.latent_size,)) for k in range(args.K)])
elif args.flow == "Radial":
    flows = SimpleFlowModel([Radial((args.latent_size,)) for k in range(args.K)])
elif args.flow == "RealNVP":
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(args.latent_size)])
    flows = []
    for i in range(args.K):
        s = nets.MLP([args.latent_size, 8, args.latent_size])
        t = nets.MLP([args.latent_size, 8, args.latent_size])
        if i % 2 == 0:
            flows += [MaskedAffineFlow(b, t, s)]
        else:
            flows += [MaskedAffineFlow(1 - b, t, s), BatchNorm()]
    flows = SimpleFlowModel(
        flows[:-1]
    )  # Remove last Batch Norm layer to allow arbitrary output

model = FlowVAE(flows).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_losses = []
train_loader, test_loader = flow_vae_datasets(args.dataset)


def train(model, epoch, beta):
    model.train()
    tr_loss = 0
    progressbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_n, (x, n) in progressbar:
        x = x.to(device)
        optimizer.zero_grad()
        rc_batch, kld = model(x)
        loss = bound(
            rc_batch, x.view(x.size(0) * x.size(1), img_dim**2), kld.sum(), beta=beta
        )
        avg_loss = loss / len(x)
        loss.backward()
        tr_loss += loss.item()
        optimizer.step()
        progressbar.update()
        if batch_n % args.log_intv == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_n * len(x),
                    len(train_loader.dataset),
                    100.0 * batch_n / len(train_loader),
                    loss.item() / len(x),
                )
            )
            print(model.test_params)
    progressbar.close()
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, tr_loss / len(train_loader.dataset)
        )
    )


def test(model, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            rc_batch, kld = model(x)
            test_loss += bound(
                rc_batch, x.view(x.size(0) * x.size(1), img_dim**2), kld.sum(), beta=1
            ).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


test_losses = []


def anneal(epoch, len_e):
    return min(1.0, 0.01 + epoch / len_e)


if __name__ == "__main__":
    if args.experiment_mode:
        min_test_losses = []
        min_test_losses.append(str(args))
        for i in range(args.runs):
            test_losses = []
            model.__init__(flows)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            if i == 0:
                seed = args.seed
            else:
                seed += 1
            torch.manual_seed(seed)
            for e in [i + 1 for i in range(args.epochs)]:
                beta = anneal(e, args.epochs)
                train(model, e, beta)
                tl = test(model, e)
                test_losses.append(tl)
            print("====> Lowest test set loss: {:.4f}".format(min(test_losses)))
            min_test_losses.append(min(test_losses))
        Series = pd.Series(min_test_losses)

        dirName = "experiments"
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        else:
            pass
        file_name = dirName + "/{}.xlsx".format(str(datetime.now()))
        file_name = file_name.replace(":", "-")
        Series.to_excel(file_name, index=False, header=None)
    else:
        for e in [i + 1 for i in range(args.epochs)]:
            beta = anneal(e, args.epochs)
            train(model, e, beta=beta)
            tl = test(model, e)
            test_losses.append(tl)
