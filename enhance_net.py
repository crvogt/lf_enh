import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, num_channels=3):
        super(UNet, self).__init__()
        self.filters = [64, 128, 256, 512, 1024]
        # comments are inputs to subsequent layer
        # 128 x 128 x 3
        self.dilation = 1
        self.padding = 1
        self.ks = 3
        self.conv1 = nn.Sequential(nn.Conv2d(num_channels, self.filters[0], kernel_size=self.ks,
                                             dilation=self.dilation, stride=1, padding=self.padding),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.filters[0], self.filters[0], kernel_size=self.ks,
                                             dilation=self.dilation, stride=1, padding=self.padding),
                                   nn.ReLU(inplace=True))

        # Now down by half 128 x 128 x 64
        self.conv3_d = nn.Sequential(nn.Conv2d(self.filters[0], self.filters[0], kernel_size=self.ks,
                                               dilation=self.dilation, stride=2, padding=self.padding),
                                     nn.ReLU(inplace=True))
        # 64 x 64 x 64
        self.conv4 = nn.Sequential(nn.Conv2d(self.filters[0], self.filters[1], kernel_size=self.ks,
                                             dilation=self.dilation, stride=1, padding=self.padding),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(self.filters[1], self.filters[1], kernel_size=self.ks,
                                             dilation=self.dilation, stride=1, padding=self.padding),
                                   nn.ReLU(inplace=True))

        # Now down by half
        # 64 x 64 x 128
        self.conv6_d = nn.Sequential(nn.Conv2d(self.filters[1], self.filters[1], kernel_size=self.ks,
                                               dilation=self.dilation,  stride=2, padding=self.padding),
                                     nn.ReLU(inplace=True))
        # 32 x 32 x 128
        self.conv7 = nn.Sequential(nn.Conv2d(self.filters[1], self.filters[2], kernel_size=self.ks,
                                             dilation=self.dilation, stride=1, padding=self.padding),
                                   nn.ReLU(inplace=True))
        self.center = nn.Sequential(nn.Conv2d(self.filters[2], self.filters[2], kernel_size=self.ks,
                                              dilation=self.dilation, stride=1, padding=self.padding),
                                   nn.ReLU(inplace=True))

        # Now start going back up
        self.convt1_u = nn.Sequential(nn.ConvTranspose2d(self.filters[2], self.filters[1], kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))
        self.convt2 = nn.Sequential(nn.Conv2d(self.filters[2], self.filters[1], kernel_size=self.ks,
                                              dilation=self.dilation, stride=1, padding=self.padding),
                                    nn.ReLU(inplace=True))
        self.convt3 = nn.Sequential(nn.Conv2d(self.filters[1], self.filters[1], kernel_size=self.ks,
                                              dilation=self.dilation, stride=1, padding=self.padding),
                                    nn.ReLU(inplace=True))

        # Up and double
        self.convt4_u = nn.Sequential(nn.ConvTranspose2d(self.filters[1], self.filters[0], kernel_size=2, stride=2),
                                      nn.ReLU(inplace=True))
        self.convt5 = nn.Sequential(nn.Conv2d(self.filters[1], self.filters[0], kernel_size=self.ks,
                                              dilation=self.dilation, stride=1, padding=self.padding),
                                    nn.ReLU(inplace=True))
        self.convt6 = nn.Sequential(nn.Conv2d(self.filters[0], self.filters[0], kernel_size=self.ks,
                                              dilation=self.dilation, stride=1, padding=self.padding),
                                    nn.ReLU(inplace=True))

        # Output sigmoid
        self.final = nn.Sequential(nn.Conv2d(self.filters[0], 3, kernel_size=self.ks,
                                             dilation=self.dilation, stride=1, padding=self.padding),
                                   nn.Sigmoid())


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3_d(conv2)

        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6_d(conv5)

        conv7 = self.conv7(conv6)
        center = self.center(conv7)

        convt1 = self.convt1_u(center)
        convt2 = self.convt2(torch.cat((convt1, conv5), dim=1))
        convt3 = self.convt3(convt2)

        convt4 = self.convt4_u(convt3)
        convt5 = self.convt5(torch.cat((convt4, conv2), dim=1))
        convt6 = self.convt6(convt5)

        return self.final(convt6)


# model = Net()
#
# test_in = torch.rand((1, 3, 128, 128))
# print(test_in.shape)
#
# test_out = model(test_in)
# print(test_out.shape)
#
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("num trainable params {}".format(trainable_params))

