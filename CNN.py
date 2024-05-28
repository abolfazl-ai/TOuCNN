import torch
import torch.nn as nn


class TopOptCNN(nn.Module):
    def __init__(self, materials_n):
        super().__init__()
        # Encoder
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        # Bottleneck
        self.b = ConvBlock(512, 1024)
        # Decoder
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        # Classifier
        self.c = nn.Conv2d(64, materials_n, kernel_size=3,
                           padding='same', padding_mode='reflect')
        self.outputs = nn.Softmax(1)

    def forward(self, inputs):
        inputs_pad, padding = pad_input(inputs)
        # Encoder
        s1, p1 = self.e1(inputs_pad)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        # Bottleneck
        b = self.b(p4)
        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        # Classifier
        c = self.c(d4)
        outputs = remove_padding(self.outputs(c), padding)
        return outputs


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3,
                               padding='same', padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3,
                               padding='same', padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0))
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


def pad_input(tensor, divisor=16):
    _, _, h, w = tensor.size()
    pad_h = 0 if h % divisor == 0 else divisor - (h % divisor)
    pad_w = 0 if w % divisor == 0 else divisor - (w % divisor)
    # pad (left, right, top, bottom)
    padding = (pad_w // 2, int(pad_w // 2 + pad_w % 2), pad_h // 2, int(pad_h / 2 + pad_h % 2))
    padded_tensor = nn.functional.pad(tensor, padding, mode='reflect')

    return padded_tensor, padding


def remove_padding(tensor, padding):
    _, _, h, w = tensor.size()
    return tensor[:, :, padding[2]:h - padding[3], padding[0]:w - padding[1]]
