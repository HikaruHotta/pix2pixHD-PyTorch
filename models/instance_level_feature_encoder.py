import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=16, n_layers=4):
        super().__init__()

        self.out_channels = out_channels
        channels = base_channels

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0), 
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]

        # Downsampling layers
        for i in range(n_layers):
            layers += [
                nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * channels),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
    
        # Upsampling layers
        for i in range(n_layers):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.layers = nn.Sequential(*layers)

    def instancewise_average_pooling(self, x, inst):
        '''
        Applies instance-wise average pooling.

        Given a feature map of size (b, c, h, w), the mean is computed for each b, c
        across all h, w of the same instance
        '''
        x_mean = torch.zeros_like(x)
        classes = torch.unique(inst, return_inverse=False, return_counts=False) # gather all unique classes present

        for i in classes:
            for b in range(x.size(0)):
                indices = torch.nonzero(inst[b:b+1] == i, as_tuple=False) # get indices of all positions equal to class i
                for j in range(self.out_channels):
                    x_ins = x[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(x_ins).expand_as(x_ins)
                    x_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat

        return x_mean

    def forward(self, x, inst):
        x = self.layers(x)
        x = self.instancewise_average_pooling(x, inst)
        return x