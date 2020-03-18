import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from skimage.draw import random_shapes
import matplotlib.pyplot as plt
import math
import time
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


class FFDNet(nn.Module):
    """
    FFDnet implementation. Note that BatchNorm layer changes its behavior while testing, so net.train() and net.eval()
    are necessary.

    """

    def __init__(self, kernel_size=(3, 3), padding=1, depth=20, in_channels=3, out_channels=3, scaling_factor=2,
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

def trim_image(img):
    """ Trim image dimensions to even numbers
    :param img : image to prepare for model (tensor, color, height, width)
    :return img: image with even dimensions (tensor, color, height, width)
    """
    # Check the height
    if img.shape[2] % 2:
        img = img[:,:,0:-1,:]
    # Check the width
    if img.shape[3] % 2:
        img = img[:,:,:,0:-1]
    return img


def prepare_plt_image(img):
    """ Convert a tensor image to a format for matplotlib
    :param img     : tensor image of the form (1, color, height, width)
    :return plt_img: image of the form (height, width, color)
    """
    plt_img = torch.squeeze(img, dim=0)
    plt_img = torch.transpose(plt_img, 1, 2)
    plt_img = plt_img.cpu().detach().numpy()
    plt_img *= 255
    plt_img = plt_img.astype(np.uint8).T
    return plt_img


def display_images(original, noisy, denoised=None, epoch=0, show_denoised=False, img_no=0):
    """ Display either 2 or 3 images
    :param original           : Origninal image
    :param noisy              : Noisy image
    :param denoised (optional): Denoised image 
    :param epoch (optional)   : Current training epoch for image title
    """
    if show_denoised:
        fig, ax = plt.subplots(1, 3)
        ax[2].imshow(denoised)
        ax[2].set_title('Retrieved')
        fig.suptitle(f'epoch={epoch + 1}')
    else:
        fig, ax = plt.subplots(2, 1)
        fig.suptitle(f'Dataset')
    fig.tight_layout()
    ax[0].imshow(original)
    ax[0].set_title('Original')
    ax[1].imshow(noisy)
    ax[1].set_title('Noisy')
    plt.show()


def create_noisy_image(img, noise_level):
    """ Adds white Gaussian noise to an image
    :param img  : Image to add noise to
    :noise_level: Amount of noise to add
    :return noisy_image: Image with added noise
    """
    return img + torch.normal(mean=0, std=noise_level/255.0, size=img.shape)


def create_datasets(use_original_noise=False, noise_level=0):
    """ Create an original/noisy pair of images. Either 'pre-noise' or generated noise
    :param use_original_noise: Set to True to use noisy image from folder
    :param noise_level:        Set use_original_noise=True and declare the level of noise here
    :return img, noisy:        original image and noisy image
    """
    img = []
    noisy = []
    data_path = '.'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_original_noise:    
            if not batch_idx % 2:
                img.append(trim_image(data))
            else:
                noisy.append(trim_image(data))
        else:
            img.append(trim_image(data))
            noisy_img = create_noisy_image(trim_image(data), noise_level)
            noisy.append(noisy_img)
    return img, noisy


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(255.0 / math.sqrt(mse))

def main():
    # Hyperparameters
    max_noise_level = 75
    noise_level = 25
    kernel_size=(3, 3)
    use_original_noise=False
    increase_noise=False
    num_epochs = 2000
    loss_func = CustomLoss()

    # Run the models
    # First check a 5x5 kernel with padding of 2
    for network_depth in [10]:
        net = FFDNet(kernel_size=(3, 3), padding=1, depth=network_depth).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        img, noisy = create_datasets(use_original_noise, noise_level)
        for image_idx, _ in enumerate(img):
            disp_img = prepare_plt_image(img[image_idx])
            disp_noisy = prepare_plt_image(noisy[image_idx])
            clean_img = img[image_idx] / 255.0
            clean_imgs = torch.cat([clean_img]).to(device)
            noisy_img = noisy[image_idx] / 255.0
            noisy_imgs = torch.cat([noisy_img]).to(device)
            sigma = torch.ones(len(clean_imgs), 1, 1, 1).to(device) * noise_level / 255.0
            psnr_list = []
            loss_list = []
            # Training data
            if image_idx + 1 < len(img):
                for i in range(num_epochs):
                    net.train()
                    optimizer.zero_grad()
                    out = net(noisy_imgs, sigma)
                    loss = loss_func(out, clean_imgs)
                    loss.backward()
                    optimizer.step()
                    disp_out = prepare_plt_image(out)
                    psnr_list.append(psnr(disp_img, disp_out))
                    loss_list.append(loss)
                    if i + 1 in [1, 2000]:
                        print(f'Epoch {i + 1}: train_loss={loss.item()}')
                        print(f'PSNR: {psnr(disp_img, disp_out)}')
                        display_images(disp_img, disp_noisy, disp_out, i, show_denoised=True, img_no=image_idx)

                epochs = np.arange(0, num_epochs, 1)
                fig, ax = plt.subplots(1, 2)
                
                ax[0].plot(epochs, psnr_list)
                ax[0].set(xlabel='Epochs', ylabel='PSNR (dB)', title='Denoising Performance')
                ax[1].plot(epochs, loss_list)
                ax[1].set(xlabel='Epochs', ylabel='Loss', title='Training Loss')
                fig.tight_layout()
                plt.show()

            # Validation data
            else:
                with torch.no_grad():
                    print(f'\n\nEvaluation of 5x5 kernel, depth: {network_depth}, sigma: {noise_level}')
                    net.eval()
                    out = net(noisy_imgs, sigma)
                    disp_out = prepare_plt_image(out)
                    loss = loss_func(out, clean_imgs)
                    print(f'Test PSNR: {psnr(disp_img, disp_out)}')
                    print(f'Test Loss: {loss}')
                    display_images(disp_img, disp_noisy, disp_out, 0, show_denoised=True, img_no=image_idx)

if __name__ == '__main__':
    main()