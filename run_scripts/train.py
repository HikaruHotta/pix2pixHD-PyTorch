from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from loss.feature_matching_loss import Loss
from models.course_to_fine_generator import GlobalGenerator
from models.course_to_fine_generator import LocalEnhancer
from models.patchgan_discriminator import MultiscaleDiscriminator
from models.instance_level_feature_encoder import Encoder
from datasets.prepare_dataset import Dataset


def lr_lambda(epoch):
    ''' Function for scheduling learning '''
    return 1. if epoch < decay_after else 1 - float(epoch - decay_after) / (epochs - decay_after)

def weights_init(m):
    ''' Function for initializing all model weights '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)

def show_tensor_images(image_tensor):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:1], nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def train(dataloader, models, optimizers, schedulers, device, epochs):
    encoder, generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    cur_step = 0
    display_step = 100

    mean_g_loss = 0.0
    mean_d_loss = 0.0

    for epoch in range(epochs):
        # Training epoch
        for (x_real, labels, insts, bounds) in tqdm(dataloader, position=0):
            x_real = x_real.to(device)
            labels = labels.to(device)
            insts = insts.to(device)
            bounds = bounds.to(device)

            with torch.cuda.amp.autocast():
                g_loss, d_loss, x_fake = loss_fn(x_real, labels, insts, bounds, encoder, generator, discriminator)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'
                      .format(cur_step, mean_g_loss, mean_d_loss))
                show_tensor_images(x_fake.to(x_real.dtype))
                show_tensor_images(x_real)
                mean_g_loss = 0.0
                mean_d_loss = 0.0
            cur_step += 1

        g_scheduler.step()
        d_scheduler.step()


if __name__ == '__main__':
    n_classes = 35                  # total number of object classes
    rgb_channels = n_features = 3
    device = 'cuda'
    train_dir = ['datasets']
    epochs = 20                    # total number of train epochs
    decay_after = 10               # number of epochs with constant lr
    lr = 0.0002
    betas = (0.5, 0.999)

    loss_fn = Loss(device=device)

    ## Phase 1: Low Resolution (1024 x 512)
    dataloader1 = DataLoader(
        Dataset(train_dir, target_width=1024, n_classes=n_classes),
        collate_fn=Dataset.collate_fn, batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
    )
    encoder = Encoder(rgb_channels, n_features).to(device).apply(weights_init)
    generator1 = GlobalGenerator(n_classes + n_features + 1, rgb_channels).to(device).apply(weights_init)
    discriminator1 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels, n_discriminators=2).to(device).apply(weights_init)

    g1_optimizer = torch.optim.Adam(list(generator1.parameters()) + list(encoder.parameters()), lr=lr, betas=betas)
    d1_optimizer = torch.optim.Adam(list(discriminator1.parameters()), lr=lr, betas=betas)
    g1_scheduler = torch.optim.lr_scheduler.LambdaLR(g1_optimizer, lr_lambda)
    d1_scheduler = torch.optim.lr_scheduler.LambdaLR(d1_optimizer, lr_lambda)

    ## Phase 2: High Resolution (2048 x 1024)
    dataloader2 = DataLoader(
        Dataset(train_dir, target_width=2048, n_classes=n_classes),
        collate_fn=Dataset.collate_fn, batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
    )
    generator2 = LocalEnhancer(n_classes + n_features + 1, rgb_channels).to(device).apply(weights_init)
    discriminator2 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels).to(device).apply(weights_init)

    g2_optimizer = torch.optim.Adam(list(generator2.parameters()) + list(encoder.parameters()), lr=lr, betas=betas)
    d2_optimizer = torch.optim.Adam(list(discriminator2.parameters()), lr=lr, betas=betas)
    g2_scheduler = torch.optim.lr_scheduler.LambdaLR(g2_optimizer, lr_lambda)
    d2_scheduler = torch.optim.lr_scheduler.LambdaLR(d2_optimizer, lr_lambda)

        # Phase 1: Low Resolution
    #######################################################################
    train(
        dataloader1,
        [encoder, generator1, discriminator1],
        [g1_optimizer, d1_optimizer],
        [g1_scheduler, d1_scheduler],
        device,
        epochs
    )


    # Phase 2: High Resolution
    #######################################################################
    # Update global generator in local enhancer with trained
    generator2.g1 = generator1.g1

    # Freeze encoder and wrap to support high-resolution inputs/outputs
    def freeze(encoder):
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        @torch.jit.script
        def forward(x, inst):
            x = F.interpolate(x, scale_factor=0.5, recompute_scale_factor=True)
            inst = F.interpolate(inst.float(), scale_factor=0.5, recompute_scale_factor=True)
            feat = encoder(x, inst.int())
            return F.interpolate(feat, scale_factor=2.0, recompute_scale_factor=True)
        return forward

    train(
        dataloader2,
        [freeze(encoder), generator2, discriminator2],
        [g2_optimizer, d2_optimizer],
        [g2_scheduler, d2_scheduler],
        device,
        epochs
    )






