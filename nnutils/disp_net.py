'''
Inverse depth prediction net.
Code based on https://github.com/ClementPinard/dispNetPytorch/
'''
import torch
import torch.nn as nn
import math
from . import net_blocks as nb

def predict_disp(in_planes):
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=False)

class DispNet(nn.Module):
    expansion = 1

    def __init__(self, batch_norm=True):
        super(DispNet, self).__init__()

        self.batch_norm = batch_norm
        self.conv1   = nb.conv2d(self.batch_norm,   3,   64, kernel_size=7, stride=2)
        self.conv2   = nb.conv2d(self.batch_norm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = nb.conv2d(self.batch_norm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = nb.conv2d(self.batch_norm, 256,  256)
        self.conv4   = nb.conv2d(self.batch_norm, 256,  512, stride=2)
        self.conv4_1 = nb.conv2d(self.batch_norm, 512,  512)
        self.conv5   = nb.conv2d(self.batch_norm, 512,  512, stride=2)
        self.conv5_1 = nb.conv2d(self.batch_norm, 512,  512)
        self.conv6   = nb.conv2d(self.batch_norm, 512, 1024, stride=2)
        self.conv6_1 = nb.conv2d(self.batch_norm,1024, 1024)

        self.deconv5 = nb.deconv2d(1024,512)
        self.deconv4 = nb.deconv2d(1025,256)
        self.deconv3 = nb.deconv2d(769,128)
        self.deconv2 = nb.deconv2d(385,64)
        self.deconv1 = nb.deconv2d(193,64)

        self.predict_disp6 = predict_disp(1024)
        self.predict_disp5 = predict_disp(1025)
        self.predict_disp4 = predict_disp(769)
        self.predict_disp3 = predict_disp(385)
        self.predict_disp2 = predict_disp(193)
        self.predict_disp1 = predict_disp(129)

        self.upsampled_disp6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_disp5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_disp4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_disp3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_disp2_to_1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.02 / n) #this modified initialization seems to work better, but it's very hacky
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        disp6       = self.predict_disp6(out_conv6)
        disp6_up    = self.upsampled_disp6_to_5(disp6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,disp6_up),1)
        disp5       = self.predict_disp5(concat5)
        disp5_up    = self.upsampled_disp5_to_4(disp5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4,out_deconv4,disp5_up),1)
        disp4       = self.predict_disp4(concat4)
        disp4_up    = self.upsampled_disp4_to_3(disp4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3,out_deconv3,disp4_up),1)
        disp3       = self.predict_disp3(concat3)
        disp3_up    = self.upsampled_disp3_to_2(disp3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,disp3_up),1)
        disp2 = self.predict_disp2(concat2)
        disp2_up    = self.upsampled_disp2_to_1(disp2)
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_conv1,out_deconv1,disp2_up),1)
        disp1       = self.predict_disp1(concat1)

        if self.training:
            #return disp1,disp2,disp3,disp4,disp5,disp6
            return disp1
        else:
            return disp1


def dispnet(path=None, batch_norm=True):
    """dispNet model architecture.

    Args:
        path : where to load pretrained network. will create a new one if not set
    """
    model = DispNet(batch_norm=batch_norm)
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model