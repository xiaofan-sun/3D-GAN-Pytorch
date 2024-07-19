import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, in_dim=(256, 144, 48), in_channels=1, out_conv_channels=512):
        super(Discriminator, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.out_dim = (int(in_dim[0] / 16), int(in_dim[1] / 16), int(in_dim[2] / 16))

        def build_conv_layer(in_channels, out_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)):
            layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU(0.03, inplace=True))
            layers.append(nn.Dropout(0.03))
            return nn.Sequential(*layers)

        self.conv_layers = nn.ModuleList([
            build_conv_layer(in_channels, conv1_channels),
            build_conv_layer(conv1_channels, conv2_channels),
            build_conv_layer(conv2_channels, conv3_channels),
            build_conv_layer(conv3_channels, out_conv_channels)
        ])
        
        self.adv = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim[0] * self.out_dim[1] * self.out_dim[2], 1),
            nn.Sigmoid(),
        )

        self.adv = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim[0] * self.out_dim[1] * self.out_dim[2], 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1),
            nn.Softmax()
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.shape[0],-1)
        x_adv = self.adv(x)
        x_aux = self.aux(x)
        return x_adv, x_aux


class Generator(nn.Module):
    def __init__(self, out_dim=(256, 144, 48), in_channels=512, out_channels=1, noise_dim=256, activation="sigmoid"):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.in_dim = (int(out_dim[0] / 16), int(out_dim[1] / 16), int(out_dim[2] / 16))
        conv1_out_channels = int(self.in_channels / 2)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, in_channels * self.in_dim[0] * self.in_dim[1] * self.in_dim[2]),
            nn.ReLU()
        )

        self.linear = nn.Linear(noise_dim, in_channels * self.in_dim[0] * self.in_dim[1] * self.in_dim[2])
        self.batch_norm_dense = nn.BatchNorm1d(in_channels * self.in_dim[0] * self.in_dim[1] * self.in_dim[2])
        self.leaky_relu_dense = nn.LeakyReLU(0.03)

        def build_conv_transpose_layer(in_channels, out_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), batch_norm=True, relu=True, dropout=True):
            layers = [nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm3d(out_channels))
            if relu:
                layers.append(nn.LeakyReLU(0.03, inplace=True))
                # layers.append(nn.ReLU(inplace=True)),
            if dropout:
                layers.append(nn.Dropout(0.03))
            return nn.Sequential(*layers)
        
        self.conv_transpose_layers = nn.ModuleList([
            build_conv_transpose_layer(in_channels, conv1_out_channels),
            build_conv_transpose_layer(conv1_out_channels, conv2_out_channels),
            build_conv_transpose_layer(conv2_out_channels, conv3_out_channels),
            build_conv_transpose_layer(conv3_out_channels, out_channels, batch_norm=False, dropout=False, relu=False)
        ])
        
        if activation == "sigmoid":
            self.out = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Tanh()

    def project(self, x):
        """
        projects and reshapes latent vector to starting volume
        :param x: latent vector
        :return: starting volume
        """
        x = self.batch_norm_dense(x)
        x = self.leaky_relu_dense(x)
        return x.view(-1, self.in_channels, self.in_dim[0], self.in_dim[1], self.in_dim[2])

    def forward(self, x, att):
        x = torch.cat([x, att])
        # x = self.linear(x)
        x = self.fc(x)
        x = self.project(x)
        for layer in self.conv_transpose_layers:
            x = layer(x)
            # print("shape",x.shape)
        return self.out(x)