import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.draw import random_shapes
from skimage.io import imsave
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import time

torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FFDBlock(nn.Module):
    """
    FFDnet building block.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, momentum=0.5, batch_norm=True,
                 activation=True):
        """
        A FFDnet layer's input channel could be different from a RBG image channel. For example, the input FFDnet layer
        takes as input the downsampling output and the noise map, so the input channel should be 4C+1=13.

        The output channel should be 4C=12, which will remain the same in each FFDnet layer.


        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param momentum: A BatchNorm2D parameter.
        :param batch_norm: For the input and output layer, batch_norm=False.
        :param activation: For the output layer, activation=False.
        """
        super(FFDBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels, momentum=momentum))
        if activation:
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CustomLoss(nn.Module):
    """
    This loss function is the Eq.(5) of the paper.

    """

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, noisy, clean):
        n_sample = noisy.size()[0]
        norms = torch.norm((noisy - clean).view(n_sample, -1), p=2, dim=1, keepdim=True) ** 2
        return torch.mean(norms) / 2


def get_psnr(noisy, clean):
    n_sample = noisy.size()[0]
    mse = nn.MSELoss()(noisy.view(n_sample, -1), clean.view(n_sample, -1))
    psnr = 20 * torch.log10(torch.tensor([1]).float().to(device) / torch.sqrt(mse))
    return torch.mean(psnr)


class FFDNet(nn.Module):
    """
    FFDnet implementation. Note that BatchNorm layer changes its behavior while testing, so net.train() and net.eval()
    are necessary.

    """

    def __init__(self, kernel_size=(3, 3), padding=1, depth=16, in_channels=3, out_channels=3, scaling_factor=2,
                 momentum=0.5):
        """

        :param kernel_size:
        :param padding:
        :param depth: The depth shows how many building blocks in a FFDnet(including the input and output layers).
        :param in_channels: The number of input channels should be 3 since we use RGB images.
        :param out_channels: The same as in_channels.
        :param scaling_factor: The default value is 2.
        :param momentum: A BatchNorm2D parameter.
        """
        super(FFDNet, self).__init__()
        self.scaling_factor = scaling_factor
        self.pixel_shuffle = nn.PixelShuffle(scaling_factor)  # Upsampling

        # FFDnet input and output channels
        ffd_in_channels = in_channels * scaling_factor ** 2 + 1  # 4C+1
        ffd_out_channels = out_channels * scaling_factor ** 2  # 4C

        # FFDnet
        layers = []
        # Input layer: Conv + ReLU
        layers.append(FFDBlock(in_channels=ffd_in_channels, out_channels=ffd_out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, batch_norm=False))
        # Middle layers: Conv + BN + ReLU
        for i in range(depth - 2):
            layers.append(FFDBlock(in_channels=ffd_out_channels, out_channels=ffd_out_channels, kernel_size=kernel_size,
                                   stride=1, padding=padding, momentum=momentum))
        # Output layer: Conv
        layers.append(FFDBlock(in_channels=ffd_out_channels, out_channels=ffd_out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, batch_norm=False, activation=False))
        # Assemble the layers into the model
        # layers.append(nn.Sigmoid())
        self.ffdnet = nn.Sequential(*layers)

    def pixel_unshuffle(self, input, upscale_factor):
        r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
        tensor of shape :math:`(*, r^2C, H, W)`.
        Authors:
            Zhaoyi Yan, https://github.com/Zhaoyi-Yan
            Kai Zhang, https://github.com/cszn/FFDNet
        Date:
            01/Jan/2019
        """
        batch_size, channels, in_height, in_width = input.size()

        out_height = in_height // upscale_factor
        out_width = in_width // upscale_factor

        input_view = input.contiguous().view(
            batch_size, channels, out_height, upscale_factor,
            out_width, upscale_factor)

        channels *= upscale_factor ** 2
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(batch_size, channels, out_height, out_width)

    def forward(self, x, sigma):
        """

        :param x: A tensor of a shape (N,4*C,W/2,H/2)
        :param sigma: A tensor of a shape (N,1,1,1)
        :return: A tensor of a shape (N,C,W,H)
        """
        # Downsample the input
        x = self.pixel_unshuffle(x, self.scaling_factor)

        # Add noise matrix to downsampled input
        m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x) * sigma
        x = torch.cat((x, m), 1)

        # Foward pass through network
        out = self.ffdnet(x)

        # Upsample the results to the clean image
        out = self.pixel_shuffle(out)
        return out


class NoisyImageDataset(Dataset):
    def __init__(self, clean_images, noise_level, labels=None):
        self.clean_images = clean_images
        self.noise_level = noise_level
        self.labels = labels

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        image = self.clean_images[idx].astype(np.float32)
        if self.labels is not None:
            sigma = self.noise_level
            noisy_image = self.labels[idx]
        else:
            if isinstance(self.noise_level, list):
                sigma = np.random.randint(self.noise_level[0], self.noise_level[1])
            else:
                sigma = self.noise_level
            noisy_image = image + np.random.normal(loc=0, scale=sigma, size=image.shape)
            noisy_image[noisy_image > 255.0] = 255.0
            noisy_image[noisy_image < 0] = 0.0
            image, noisy_image = image / 255.0, noisy_image / 255.0
        noise_level = sigma / 255.0
        return image, noisy_image, noise_level


@click.command()
@click.option('--depth', default=16, help='Network depth.', type=int)
@click.option('--sigma', help='Noise level.', type=int, default=5)
@click.option('--dataset_path', help='Dataset path.', type=str, default='data/synthetic/all_sigma_5.npz')
@click.option('--store_root', help='Store folder.', type=str, default='results/sigma_5')
@click.option('--learning_rate', help='Learning rate.', type=float, default=0.001)
@click.option('--max_epochs', help='Max training epochs.', type=int, default=10000)
@click.option('--min_val_loss', help='Early stopping validation loss.', type=float, default=0.0)
@click.option('--train_batch_size', help='Train batch size.', type=int, default=20)
@click.option('--kernel_size', help='Kernel size.', type=int, default=3)
def train(depth, sigma, dataset_path, store_root, learning_rate, max_epochs, min_val_loss, train_batch_size,
          kernel_size):
    start_time = time.time()
    if not os.path.exists(store_root):
        os.makedirs(store_root)
    padding = (kernel_size - 1) // 2
    figure_root = os.path.join(store_root, 'figures')
    model_path = os.path.join(store_root, 'model.pt')
    result_path = os.path.join(store_root, 'result.npz')

    data = np.load(dataset_path)
    train_x, train_y, val_x, val_y, test_x, test_y = data['train_x'], data['train_y'], data['val_x'], data['val_y'], \
                                                     data['test_x'], data['test_y']
    train_x, train_y, val_x, val_y, test_x, test_y = train_x / 255.0, train_y / 255.0, val_x / 255.0, val_y / 255.0, test_x / 255.0, test_y / 255.0

    noise_range = sigma
    train_noise_level = sigma
    val_noise_level = sigma
    test_noise_level = sigma

    net = FFDNet(depth=depth, kernel_size=(kernel_size, kernel_size), padding=padding).to(device)
    loss_func = CustomLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_dataset = NoisyImageDataset(train_x, noise_range, labels=train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    train_x_tensor = torch.tensor(train_x).float().to(device)
    train_y_tensor = torch.tensor(train_y).float().to(device)
    train_sigma = (torch.ones(len(train_y), 1, 1, 1).to(device) * train_noise_level / 255.0).float()

    val_x_tensor = torch.tensor(val_x).float().to(device)
    val_y_tensor = torch.tensor(val_y).float().to(device)
    val_sigma = (torch.ones(len(val_y), 1, 1, 1).to(device) * val_noise_level / 255.0).float()

    test_x_tensor = torch.tensor(test_x).float().to(device)
    test_y_tensor = torch.tensor(test_y).float().to(device)
    test_sigma = (torch.ones(len(test_y), 1, 1, 1).to(device) * test_noise_level / 255.0).float()

    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        for image, noisy_image, noise_level in train_loader:
            net.train()
            image, noisy_image, noise_level = image.to(device).float(), noisy_image.to(device).float(), noise_level.to(
                device).float()
            sigma = noise_level.reshape(-1, 1, 1, 1).float()
            optimizer.zero_grad()
            out = net(noisy_image, sigma)
            loss = loss_func(out, image)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            net.eval()
            val_out = net(val_y_tensor, val_sigma)
            val_loss = loss_func(val_out, val_x_tensor).item()
            train_out = net(train_y_tensor, train_sigma)
            train_loss = loss_func(train_out, train_x_tensor).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss <= min_val_loss:
                print(f'Early stopping: epoch={epoch}, train_loss={train_loss}, val_loss={val_loss}')
                break
        if epoch % 200 == 0:
            print(f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}')
    torch.save(net.state_dict(), model_path)
    train_losses = np.array(train_losses).flatten()
    val_losses = np.array(val_losses).flatten()

    with torch.no_grad():
        net.eval()
        test_out = net(test_y_tensor, test_sigma)
        test_loss = loss_func(test_out, test_x_tensor).item()
        test_psnr = get_psnr(test_out, test_x_tensor).item()
        print(f'test_psnr={test_psnr}')
    end_time = time.time()
    running_time = end_time - start_time
    np.savez_compressed(result_path, train_losses=train_losses, val_losses=val_losses, test_loss=test_loss,
                        test_psnr=test_psnr, running_time=running_time)


if __name__ == '__main__':
    train()
