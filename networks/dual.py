# camera-ready

import torch.nn as nn
import torch.nn.functional as F
import torch


class LearnableSigmoid(nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def forward(self, input):
        return (2 / (1 + torch.exp(-self.weight * input)))-1


class LearnableSoftmax(nn.Module):
    def __init__(self, ):
        super(LearnableSoftmax, self).__init__()
        self.epsilon = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.epsilon.data.fill_(1.0)

    def forward(self, input):
        return F.softmax(input / torch.exp(self.epsilon), dim=-1), torch.exp(self.epsilon)


class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()

        filter_num_list = [4096, 2048, 1024, 1]

        self.fc1 = nn.Linear(24576, filter_num_list[0])
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(filter_num_list[0], filter_num_list[1])
        self.fc3 = nn.Linear(filter_num_list[1], filter_num_list[2])
        self.fc4 = nn.Linear(filter_num_list[2], filter_num_list[3])

        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    # m.bias.data.copy_(1.0)
                    m.bias.data.zero_()


    def forward(self, x):

        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.fc4(x)
        return x


class OutputDiscriminator(nn.Module):
    def __init__(self, ):
        super(OutputDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(2, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x


class UDual(nn.Module):
    def __init__(self, ):
        super(UDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(3, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(289, 3)
        self.learnsoftmax = LearnableSoftmax()
         # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x, epsilon = self.learnsoftmax(x)
        return x, epsilon


class WDual(nn.Module):
    def __init__(self, ):
        super(WDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(3, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(289, 3)
        self.learnsigmoid = LearnableSigmoid()
         # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.learnsigmoid(x)
        return x

