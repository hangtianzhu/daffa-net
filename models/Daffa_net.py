import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class Lap_Pyramid_Conv(nn.Module):
    """Laplacian Pyramid Construction Module"""
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()
        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            # Ensure up and current have the same size
            if up.size(2) != current.size(2) or up.size(3) != current.size(3):
                up = F.interpolate(up, size=(current.size(2), current.size(3)), mode='bilinear', align_corners=False)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            # Ensure up and level have the same size
            if up.size(2) != level.size(2) or up.size(3) != level.size(3):
                up = F.interpolate(up, size=(level.size(2), level.size(3)), mode='bilinear', align_corners=False)
            image = up + level
        return image


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    """Patch to Image Unembedding"""
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    """Selective Kernel Fusion Module"""
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class convforlap_block(nn.Module):
    """Convolutional Block for Laplacian Pyramid"""
    def __init__(self, dim=3):
        super(convforlap_block, self).__init__()
        self.norm1 = nn.BatchNorm2d(3)
        self.norm2 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=7, padding=3, groups=1, dilation=1,
                                  padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=5, padding=4, groups=1, dilation=2,
                                  padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=3, padding=5, groups=1, dilation=5,
                                 padding_mode='reflect')

        self.pa = nn.Sequential(
            nn.Conv2d(3, 3, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(3, 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(24, 24, 1),
            nn.GELU(),
            nn.Conv2d(24, 3, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x
        identity = x
        x = self.pa(x) * x
        x = identity + x
        return x


class DENet(nn.Module):
    """Dual Branch Enhancement Network"""
    def __init__(self, num_high=3, gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.lapblock = convforlap_block(dim=3)
        self.patch_split1_twobbranch = PatchUnEmbed(patch_size=2, out_chans=3, embed_dim=3)
        self.fusion = sfsfuion_lop(dim=3)

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)
        pyrs_to_recon = []
        
        # Level 1
        pyrs[-1] = self.lapblock(pyrs[-1])
        pyrs_to_recon.append(pyrs[-1])
        pyrs[-1] = self.patch_split1_twobbranch(pyrs[-1])
        if pyrs[-2].size(2) != pyrs[-1].size(2) or pyrs[-2].size(3) != pyrs[-1].size(3):
            pyrs[-1] = F.interpolate(pyrs[-1], size=(pyrs[-2].size(2), pyrs[-2].size(3)), mode='bilinear', align_corners=False)
        pyrs[-2] = self.fusion([pyrs[-2], pyrs[-1]])

        # Level 2
        pyrs[-2] = self.lapblock(pyrs[-2])
        pyrs_to_recon.append(pyrs[-2])
        pyrs[-2] = self.patch_split1_twobbranch(pyrs[-2])
        if pyrs[-3].size(2) != pyrs[-2].size(2) or pyrs[-3].size(3) != pyrs[-2].size(3):
            pyrs[-2] = F.interpolate(pyrs[-2], size=(pyrs[-3].size(2), pyrs[-3].size(3)), mode='bilinear', align_corners=False)
        pyrs[-3] = self.fusion([pyrs[-3], pyrs[-2]])

        # Level 3
        pyrs[-3] = self.lapblock(pyrs[-3])
        pyrs_to_recon.append(pyrs[-3])
        pyrs[-3] = self.patch_split1_twobbranch(pyrs[-3])
        if pyrs[-4].size(2) != pyrs[-3].size(2) or pyrs[-4].size(3) != pyrs[-3].size(3):
            pyrs[-3] = F.interpolate(pyrs[-3], size=(pyrs[-4].size(2), pyrs[-4].size(3)), mode='bilinear', align_corners=False)
        pyrs[-4] = self.fusion([pyrs[-4], pyrs[-3]])

        # Level 4
        pyrs[-4] = self.lapblock(pyrs[-4])
        pyrs_to_recon.append(pyrs[-4])

        out = self.lap_pyramid.pyramid_recons(pyrs_to_recon)
        return out


class MixFocusedAttentionModule(nn.Module):
    """Mixed Structure Block with Multi-scale Convolution and Attention"""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, dilation=1, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=4, groups=dim, dilation=2, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=5, groups=dim, dilation=5, padding_mode='reflect')

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*3, dim*3, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim*3, dim*3, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.rpa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        # Local Pixel Attention
        self.lpa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Tanh()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x1 = self.lpa(x) * x
        x2 = self.rpa(x)
        x2 = x2 * x2
        x2 = x2 * x
        x3 = self.rpa(self.rpa(x) * x) * x
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.ca(x) * x
        x = self.mlp2(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    """Basic Layer composed of multiple MixFocusedAttentionModule"""
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            MixFocusedAttentionModule(dim=dim) for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class sfsfuion_lop(nn.Module):
    """Scale Fusion Module for Laplacian Octave Path"""
    def __init__(self, dim, height=3, reduction=8):
        super(sfsfuion_lop, self).__init__()
        self.dwtconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, padding_mode="reflect")
        )
        self.conv3 = nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1)
        self.dialconv = nn.Conv2d(dim, dim, kernel_size=7, dilation=7, stride=2)
        self.pa = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.norm1 = nn.BatchNorm2d(3)

    def forward(self, in_feats):
        fb = self.dwtconv(in_feats[1])
        fcat = torch.cat([fb, in_feats[0]], dim=1)
        fcat = self.conv3(fcat)
        fcat = self.pa(fcat)
        fadd = torch.add(fb, in_feats[0])
        fatt = fcat * fadd
        out = in_feats[0] + fatt
        return out


class sfsfuion(nn.Module):
    """Scale Fusion Module"""
    def __init__(self, dim, height=3, reduction=8):
        super(sfsfuion, self).__init__()
        self.dwtconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, padding_mode="reflect")
        )
        self.conv3 = nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1)
        self.dialconv = nn.Conv2d(dim, dim, kernel_size=7, dilation=7, stride=2)
        self.pa = nn.Sequential(
            nn.Conv2d(dim*2, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.norm1 = nn.BatchNorm2d(3)

    def forward(self, in_feats):
        fb = self.dwtconv(in_feats[1])
        fcat = torch.cat([fb, in_feats[0]], dim=1)
        fcat = self.conv3(fcat)
        fcat = self.pa(fcat)
        fadd = torch.add(fb, in_feats[0])
        fatt = fcat * fadd
        out = in_feats[0] + fatt
        return out


class sfsfuion_last(nn.Module):
    """Final Scale Fusion Module with Learnable Parameter"""
    def __init__(self, dim, height=3, reduction=8):
        super(sfsfuion_last, self).__init__()
        self.dwtconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, padding_mode="reflect")
        )
        self.conv3 = nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1)
        self.dialconv = nn.Conv2d(dim, dim, kernel_size=7, dilation=7, stride=2)
        self.pa = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.norm1 = nn.BatchNorm2d(3)
        self.a = nn.Parameter(torch.ones(1))  # Learnable parameter

    def forward(self, in_feats):
        fb = self.dwtconv(in_feats[1])
        fcat = torch.cat([fb, in_feats[0]], dim=1)
        fcat = self.conv3(fcat)
        fcat = self.pa(fcat)
        fadd = torch.add(fb, in_feats[0])
        fatt = fcat * fadd
        out = in_feats[0] + self.a * fatt
        return out


class daffa_net(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[2, 2, 4, 2, 2]):
        super(daffa_net, self).__init__()
        self.lap_pyramid = Lap_Pyramid_Conv(3, 3)
        self.patch_size = 4

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # Backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # Merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
            
        # Channel adaptation layers
        self.up_channel24 = nn.Conv2d(3, 24, kernel_size=3, padding=1)
        self.up_channel48 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.up_channel96 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        
        # Scale fusion modules
        self.csfusion24 = sfsfuion(dim=24)
        self.csfusion48 = sfsfuion(dim=48)
        self.csfusion96 = sfsfuion(dim=96)
        self.csfusion = sfsfuion_lop(dim=3)
        self.size_padding = PatchUnEmbed(patch_size=2, out_chans=3, embed_dim=3)
        self.fusion = sfsfuion_last(dim=4)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)
        # Ensure x and pyrs[0] have the same size
        if x.size(2) != pyrs[0].size(2) or x.size(3) != pyrs[0].size(3):
            pyrs[0] = F.interpolate(pyrs[0], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.patch_embed(x)
        x = self.csfusion24([x, pyrs[0]])
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        # Ensure x and pyrs[1] have the same size
        if x.size(2) != pyrs[1].size(2) or x.size(3) != pyrs[1].size(3):
            pyrs[1] = F.interpolate(pyrs[1], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.csfusion48([x, pyrs[1]])
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        # Ensure x and pyrs[2] have the same size
        if x.size(2) != pyrs[2].size(2) or x.size(3) != pyrs[2].size(3):
            pyrs[2] = F.interpolate(pyrs[2], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.csfusion96([x, pyrs[2]])
        x = self.layer3(x)
        # Ensure pyrs[3] and x have the same size
        if pyrs[3].size(2) != x.size(2) or pyrs[3].size(3) != x.size(3):
            pyrs[3] = self.size_padding(pyrs[3])
            if pyrs[3].size(2) != x.size(2) or pyrs[3].size(3) != x.size(3):
                pyrs[3] = F.interpolate(pyrs[3], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.csfusion96([x, pyrs[3]])
        x = self.layer3(x)
        x = self.patch_split1(x)
        # Ensure x and skip2 have the same size
        if x.size(2) != skip2.size(2) or x.size(3) != skip2.size(3):
            skip2 = F.interpolate(skip2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.fusion1([x, self.skip2(skip2)]) + x

        x = self.layer4(x)
        x = self.patch_split2(x)
        # Ensure x and skip1 have the same size
        if x.size(2) != skip1.size(2) or x.size(3) != skip1.size(3):
            skip1 = F.interpolate(skip1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        two_branchx = x
        denet = DENet().to(x.device)
        two_branchx = denet(two_branchx)
        feat = self.forward_features(x)
        # Ensure feat and two_branchx have the same size
        if feat.size(2) != two_branchx.size(2) or feat.size(3) != two_branchx.size(3):
            two_branchx = F.interpolate(two_branchx, size=(feat.size(2), feat.size(3)), mode='bilinear', align_corners=False)
        feat = self.fusion([feat, two_branchx])
        
        # Transmission map guided dehazing
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x





def Daffa_net():
    return daffa_net(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[2, 2, 4, 2, 2])




if __name__ == '__main__':
    model = Daffa_net().cuda(0)
    x = torch.randn(1, 3, 620, 460).cuda(0)
    x = model(x)
    print(x.shape)
    