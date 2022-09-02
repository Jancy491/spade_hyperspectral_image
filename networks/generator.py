"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G="spectralspadesyncbatch3x3")
        parser.add_argument(
            "--num_upsampling_layers",
            choices=("normal", "more", "most", "no"),
            default="no",
            # default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator",
        )

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        # encode
        # self.s2_convin = nn.Conv2d(self.opt.semantic_nc, 256, 3, stride=1,padding=1)
        # self.s2_convmiddle = nn.Conv2d(256, 256, 4, stride=4,padding=0)
        # self.s2_convout = nn.Conv2d(256, 128, 2, stride=2,padding=0)
        # self.s2_encode = nn.Sequential(self.s2_convin,_Residual_Block(),self.s2_convmiddle,_Residual_Block(),self.s2_convout)
        # self.ref_convin = nn.Conv2d(2, 256, 3, stride=1,padding=1)
        # self.ref_convmiddle = nn.Conv2d(256, 256, 4, stride=4,padding=0)
        # self.ref_convout = nn.Conv2d(256, 128, 2, stride=2,padding=0)
        # self.ref_encode = nn.Sequential(self.ref_convin,_Residual_Block(),self.ref_convmiddle,_Residual_Block(),self.ref_convout)
        # self.cat_conv = nn.Conv2d(256,16 * nf,3,1,1)

        final_nc = nf      

        if opt.num_upsampling_layers == "most":
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        # note siman : change 2 to 4 below
        self.conv_img = nn.Conv2d(final_nc, 4, 3, padding=1) 

        self.up = nn.Upsample(scale_factor=4)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == "normal":
            num_up_layers = 5
        elif opt.num_upsampling_layers == "more":
            num_up_layers = 6
        elif opt.num_upsampling_layers == "most":
            num_up_layers = 7
        elif opt.num_upsampling_layers == "no":
            num_up_layers = 0
        else:
            raise ValueError(
                "opt.num_upsampling_layers [%s] not recognized"
                % opt.num_upsampling_layers
            )

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, ref_image=None, z=None):
        seg = input
        # seg = torch.cat([input,ref_image],dim=1)
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(
                    input.size(0),
                    self.opt.z_dim,
                    dtype=torch.float32,
                    device=input.get_device(),
                )
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
            # s2_feat = self.s2_encode(input)
            # ref_feat = self.ref_encode(ref_image)
            # feat = torch.cat([s2_feat,ref_feat],axis=1)
            # x = self.cat_conv(feat)

        x = self.head_0(x, seg)
        if self.opt.num_upsampling_layers != "no":
            x = self.up(x)
        x = self.G_middle_0(x, seg)

        if (
            self.opt.num_upsampling_layers == "more"
            or self.opt.num_upsampling_layers == "most"
        ):
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        if self.opt.num_upsampling_layers != "no":
            x = self.up(x)
        x = self.up_0(x, seg)
        if self.opt.num_upsampling_layers != "no":
            x = self.up(x)
        x = self.up_1(x, seg)
        if self.opt.num_upsampling_layers != "no":
            x = self.up(x)
        x = self.up_2(x, seg)
        if self.opt.num_upsampling_layers != "no":
            x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--resnet_n_downsample",
            type=int,
            default=4,
            help="number of downsampling layers in netG",
        )
        parser.add_argument(
            "--resnet_n_blocks",
            type=int,
            default=9,
            help="number of residual blocks in the global generator network",
        )
        parser.add_argument(
            "--resnet_kernel_size",
            type=int,
            default=3,
            help="kernel size of the resnet block",
        )
        parser.add_argument(
            "--resnet_initial_kernel_size",
            type=int,
            default=7,
            help="kernel size of the first convolution",
        )
        parser.set_defaults(norm_G="instance")
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = (
            opt.label_nc
            + (1 if opt.contain_dontcare_label else 0)
            + (0 if opt.no_instance else 1)
        )

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [
            nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
            norm_layer(
                nn.Conv2d(
                    input_nc,
                    opt.ngf,
                    kernel_size=opt.resnet_initial_kernel_size,
                    padding=0,
                )
            ),
            activation,
        ]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [
                norm_layer(
                    nn.Conv2d(
                        opt.ngf * mult,
                        opt.ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),
                activation,
            ]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [
                ResnetBlock(
                    opt.ngf * mult,
                    norm_layer=norm_layer,
                    activation=activation,
                    kernel_size=opt.resnet_kernel_size,
                )
            ]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [
                norm_layer(
                    nn.ConvTranspose2d(
                        nc_in,
                        nc_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                ),
                activation,
            ]
            mult = mult // 2

        # final output conv
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.relu = nn.LeakyReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, identity_data)
        return output


class EDSRGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()

        self.conv_input = nn.Conv2d(
            in_channels=9,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.residual = self.make_layer(_Residual_Block, 32)

        self.conv_mid = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv_output = nn.Conv2d(
            in_channels=256,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, z=None):
        out = self.conv_input(x)
        out = self.conv_mid(self.residual(out))
        out = self.conv_output(out)
        return out
