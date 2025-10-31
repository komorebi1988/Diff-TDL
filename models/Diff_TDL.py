import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class EfficientNetB3(nn.Module):
    def __init__(self):
        super(EfficientNetB3, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 4)
        )
        layers = [MBConvBlock(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(MBConvBlock(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = F.max_pool2d(F.relu(out1), kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./cache/efficientnet-b3-5fb5a3c3.pth'), strict=False)


class MBConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(MBConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, n_input=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(pretrained=False, layers=[3, 4, 6, 3], backbone='resnet50', n_input=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, layers, n_input=n_input, **kwargs)

    pretrain_dict = model_zoo.load_url(model_urls[backbone])
    try:
        model.load_state_dict(pretrain_dict, strict=False)
    except:
        print("loss conv1")
        model_dict = {}
        for k, v in pretrain_dict.items():
            if k in pretrain_dict and 'conv1' not in k:
                model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
    print("load pretrain success")
    return model


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, n_input=3):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def base_forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out


class AttentionFusion(nn.Module):
    def __init__(self, in_channels_list, reduction=16):
        super(AttentionFusion, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),  
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),  
                nn.Sigmoid() 
            )
            for in_channels in in_channels_list
        ])

    def forward(self, feature_maps):
        target_size = feature_maps[0].shape[2:] 
        weighted_features = []
        for i, feature_map in enumerate(feature_maps):
            resized_feature_map = F.interpolate(feature_map, size=target_size, mode='bilinear', align_corners=True)
            attention = self.attention_layers[i](resized_feature_map)
            weighted_feature = resized_feature_map * attention
            weighted_features.append(weighted_feature)

        return torch.cat(weighted_features, dim=1)




class DFCCBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7, use_rdp=True, use_fft=False, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.use_rdp = use_rdp
        self.use_fft = use_fft

        # ---- Gaussian depthwise conv for pseudo-denoise ----
        k = 5
        sigma = 1.2
        g = self._gaussian_kernel(k, sigma)  
        self.register_buffer("g_kernel", g)  
        self.depthwise = nn.Conv2d(channel, channel, kernel_size=k, padding=k//2,
                                   groups=channel, bias=False)
        with torch.no_grad():
            self.depthwise.weight.copy_(g.expand(channel, 1, k, k))
        self.depthwise.weight.requires_grad_(False) 


        # ---- USM kernel for RDP ----
        if use_rdp:
            usm = torch.tensor([[[[0,-1,0],[-1,5,-1],[0,-1,0]]]], dtype=torch.float32)  # 3x3 unsharp mask
            self.register_buffer("usm_kernel", usm)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()

        # ---- Spatial Attention ----
        in_s = 5              
        if use_rdp: in_s += 1 
        padding = kernel_size // 2
        self.conv_s = nn.Conv2d(in_s, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    @staticmethod
    def _gaussian_kernel(ks, sigma):
        ax = torch.arange(ks) - (ks - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kern = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kern /= kern.sum()
        return kern.view(1, 1, ks, ks)

    def _pseudo_denoise(self, x):
        return self.depthwise(x)

    def _rdp(self, x):
        # USM(Gσ(x)) with groups conv (apply to channel-mean to减负担，再广播)
        x_mean = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x_mean, self.g_kernel, padding=self.g_kernel.shape[-1]//2)
        usm = F.conv2d(gx, self.usm_kernel, padding=1)
        return (x - usm)  # broadcast along channels

    @staticmethod
    def _grad(x):
        sobel_x = x.new_tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]])
        sobel_y = x.new_tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]])
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-12)
        return gx, gy, mag

    def _coherence(self, x_mean):
        gx, gy, _ = self._grad(x_mean)
        J11 = F.avg_pool2d(gx*gx, 3, 1, 1)
        J22 = F.avg_pool2d(gy*gy, 3, 1, 1)
        J12 = F.avg_pool2d(gx*gy, 3, 1, 1)
        trace = J11 + J22
        det = J11*J22 - J12*J12
        tmp = torch.sqrt(torch.clamp(trace*trace - 4*det, min=0.0))
        lam1 = (trace + tmp) * 0.5
        lam2 = (trace - tmp) * 0.5
        coh = (lam1 - lam2) / (lam1 + lam2 + 1e-6)
        return coh

    def _phase_consistency(self, x_mean):
        gx, gy, gmag = self._grad(x_mean)
        lap = F.conv2d(x_mean, x_mean.new_tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]]), padding=1).abs()
        pc = gmag / (gmag + lap + 1e-6)
        return pc

    def _norm(self, t):
        mean = t.mean(dim=(2,3), keepdim=True)
        std  = t.std(dim=(2,3), keepdim=True) + self.eps
        return (t - mean) / std

    def forward(self, x, cond:dict=None):
        residual = x
        B, C, H, W = x.shape

        with torch.no_grad():
            gx = self._pseudo_denoise(x.detach())
            r  = (x - gx).abs()
            r_mean = r.mean(dim=1, keepdim=True)

        rdp_mean = None
        if self.use_rdp:
            with torch.no_grad():
                rdp = self._rdp(x.detach()).abs()
                rdp_mean = rdp.mean(dim=1, keepdim=True)

        ca = self.se(self.avgpool(torch.nan_to_num(x))) \
        + self.se(self.maxpool(torch.nan_to_num(x))) \
        + self.se(self.avgpool(torch.nan_to_num(r)))
        if self.use_rdp:
            ca = ca + self.se(self.avgpool(torch.nan_to_num(rdp if self.use_rdp else x.new_zeros(()))))

        ca = torch.nan_to_num(ca, nan=0.0, posinf=0.0, neginf=0.0)
        ca = self.sigmoid_c(ca)
        x  = x * ca

        # ---------------- Spatial Attention ----------------
        max_c, _ = torch.max(x, dim=1, keepdim=True)
        avg_c    = torch.mean(x, dim=1, keepdim=True)
        x_mean   = avg_c
        
        with torch.no_grad():
            coh = self._coherence(torch.nan_to_num(x_mean))
            pc  = self._phase_consistency(torch.nan_to_num(x_mean))

            s_list = [ self._norm(max_c), self._norm(avg_c),
                    self._norm(r_mean), self._norm(coh), self._norm(pc) ]
            if self.use_rdp and rdp_mean is not None:
                s_list.append(self._norm(rdp_mean))
            s_in = torch.cat([torch.nan_to_num(t) for t in s_list], dim=1)

        sa = self.sigmoid_s(self.conv_s(s_in))
        sa = torch.nan_to_num(sa, nan=0.0, posinf=0.0, neginf=0.0)

        out = x * sa
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out + residual



class TIFD_CBAM(ResNet50):
    def __init__(self, nclass, aux=False, n_input=3, **kwargs):
        super(TIFD_CBAM, self).__init__(pretrained=True, n_input=n_input)
        self.num_class = nclass
        self.aux = aux
        self.head = _DAHead(2048 + 256 + 512 + 1024 + 2048, self.num_class, aux, **kwargs)

        self.effic = EfficientNetB3()

        self.cbam_c4    = DFCCBAM(2048, use_rdp=True)  
        self.cbam_out5v = DFCCBAM(2048, use_rdp=False)
        self.cbam_out4h = DFCCBAM(1024, use_rdp=True)  
        self.cbam_out3h = DFCCBAM(512,  use_rdp=True)   
        self.cbam_out2h = DFCCBAM(256,  use_rdp=False)   

        self.attention_fusion = AttentionFusion([2048, 256, 512, 1024, 2048])

    def forward(self, x):
        size = x.size()[2:]
        input_ = x.clone()
        feature_map, _ = self.base_forward(input_)
        c1, c2, c3, c4 = feature_map

        out2h, out3h, out4h, out5v = self.effic(x)


        c4 = self.cbam_c4(c4)
        out5v = self.cbam_out5v(out5v)
        out4h = self.cbam_out4h(out4h)
        out3h = self.cbam_out3h(out3h)
        out2h = self.cbam_out2h(out2h)

        combined_features = self.attention_fusion([c4, out2h, out3h, out4h, out5v])

        outputs = []
        x = self.head(combined_features)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)  
        outputs.append(x0)

        return x0



class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


def get_TIFD(backbone='resnet50', pretrained_base=True, nclass=1, n_input=3, **kwargs):
    model = TIFD_CBAM(nclass, backbone=backbone,
                               pretrained_base=pretrained_base,
                               n_input=n_input,
                               **kwargs)
    return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 512, 512)
    model = get_TIFD(n_input=3)
    edge, outputs = model(img)
    print(outputs.shape)
    print(edge.shape)
