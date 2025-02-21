# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pdb


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, M, num_components):
        """
        Define a Mixture of Gaussians prior distribution.

        Parameters:
        M: [int]
           Dimension of the latent space.
        num_components: [int]
           Number of components in the mixture.
        """
        super(MixtureOfGaussiansPrior, self).__init__()
        self.M = M
        self.num_components = num_components

        self.mixture_weights = nn.Parameter(torch.ones(num_components) / num_components, requires_grad=True)
        self.means = nn.Parameter(torch.randn(num_components, M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(num_components, M), requires_grad=True)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mixture_distribution = td.Categorical(logits=self.mixture_weights)
        component_distribution = td.Independent(td.Normal(loc=self.means, scale=self.stds), 1)
        return td.MixtureSameFamily(mixture_distribution, component_distribution)    

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net, learnable_variance=True):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters:
        decoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M)` as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension `(batch_size, feature_dim1, feature_dim2)`.
        learnable_variance: [bool]
           Whether to learn the variance for each pixel or use a fixed variance.
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.learnable_variance = learnable_variance
        if learnable_variance:
            self.log_var = nn.Parameter(torch.zeros(28, 28), requires_grad=True)
        else:
            self.log_var = torch.zeros(28, 28)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        mean = self.decoder_net(z)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(0.5 * self.log_var)), 2)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x)
        z = q.rsample()  # Reparameterization trick
        log_prob = self.decoder(z).log_prob(x)
        
        if isinstance(self.prior, MixtureOfGaussiansPrior):
            # Compute log probability under the MoG prior based on the samples z
            log_prior = self.prior().log_prob(z) #E_{z~q(z|x)}[log p(z)]
            # Compute the KL divergence
            # This is the difference between the expectation (w.r.t samples z from the approximation posterior) 
            # of the log probability of the samples under the approximate posterior and the prior
            kl_div = q.log_prob(z) - log_prior #E_{z~q(z|x)}[log q(z|x)] - E_{z~q(z|x)}[log p(z)]
            # This calculation can't be done using the td.kl_divergence function, as it requires the two distributions to be of the same type
        else:
            # Compute the KL divergence for Gaussian prior
            kl_div = td.kl_divergence(q, self.prior()) # KL(q(z|x) || p(z))
        
        elbo = torch.mean(log_prob - kl_div, dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        mean = self.decoder(z).base_dist.loc # Sample from the mean of the distribution p(x|z)
        return mean
        # return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def evaluate_test_elbo(model, data_loader, device): # Exercise 1.5.1
    """
    Evaluate the ELBO on the test set.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader for the test set.
    device: [torch.device]
        The device to use for evaluation.
    """
    model.eval()
    total_elbo = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo = model.elbo(x)
            total_elbo += elbo.item() * x.size(0)  # Multiply by batch size to get the total ELBO for the batch

    average_elbo = total_elbo / len(data_loader.dataset)
    return average_elbo

def plot_latent_space(model, data_loader, device, latent_dim): # Exercise 1.5.2
    """
    Plot samples from the approximate posterior and color them by their correct class label.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader for the test set.
    device: [torch.device]
        The device to use for evaluation.
    latent_dim: [int]
        The dimension of the latent space.
    """
    model.eval()
    all_z = []
    all_labels = []
    with torch.no_grad():
        for x, labels in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()  # Sample from the approximate posterior q(z|x)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if latent_dim > 2:
        pca = PCA(n_components=2)
        all_z = pca.fit_transform(all_z)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.savefig('latent_space.png')


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'evaluate', 'plot'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=10, metavar='N', help='number of components in the mixture of Gaussians prior (default: %(default)s)')
    parser.add_argument('--prior-type', type=str, default='gaussian', choices=['gaussian', 'mog'], help='type of prior distribution (default: %(default)s)')
    parser.add_argument('--binary', action='store_true', help='whether to use binarized MNIST (default: %(default)s)')
    parser.add_argument('--learnable-variance', action='store_true', help='whether to learn the variance for each pixel (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    if args.binary:
        # Load MNIST as binarized at 'threshold' and create data loaders
        thresshold = 0.5
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
    else:
        # Load MNIST as grayscale and create data loaders
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
        
    # Define prior distribution
    M = args.latent_dim
    num_components = args.num_components
    if args.prior_type == 'gaussian':
        prior = GaussianPrior(M)
    else:
        prior = MixtureOfGaussiansPrior(M, num_components)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    if args.binary:
        decoder = BernoulliDecoder(decoder_net)
    else:
        decoder = GaussianDecoder(decoder_net, args.learnable_variance)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Evaluate ELBO on the test set
        average_elbo = evaluate_test_elbo(model, mnist_test_loader, args.device)
        print(f"Average ELBO on the test set: {average_elbo:.4f}")
    
    elif args.mode == 'plot':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Plot latent space
        plot_latent_space(model, mnist_test_loader, args.device, args.latent_dim)
