import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from rfs_learn import GaussianNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNet(nn.Module):
    def __init__(self, resnet_type):
        super(ResNet, self).__init__()
        orig_model = ''
        if resnet_type == 'resnet18':
            orig_model = models.resnet18(pretrained=True)
        elif resnet_type == 'resnet34':
            orig_model = models.resnet34(pretrained=True)
        elif resnet_type == 'resnet50':
            orig_model = models.resnet50(pretrained=True)
        elif resnet_type == 'resnet101':
            orig_model = models.resnet101(pretrained=True)
        elif resnet_type == 'resnet152':
            orig_model = models.resnet152(pretrained=True)
        self.noise = GaussianNoise()
        self.drop = nn.Dropout2d(0.5).to(device)
        if resnet_type in ['resnet18', 'resnet34']:
            self.bn = nn.BatchNorm2d(512,momentum=0.9).to(device)
            self.bn2 = nn.BatchNorm1d(256,momentum=0.9).to(device)
            self.fc = nn.Linear(512, 256).to(device)
            self.fc2 = nn.Linear(256, 100).to(device)
        else:
            self.bn = nn.BatchNorm2d(2048, momentum=0.9).to(device)
            self.bn2 = nn.BatchNorm1d(512, momentum=0.9).to(device)
            self.fc = nn.Linear(2048, 512).to(device)
            self.fc2 = nn.Linear(512, 100).to(device)
        self.bn3 = nn.BatchNorm1d(100,momentum=0.9).to(device)
        self.orig = nn.Sequential(*(list(orig_model.children())[:-1])).to(device)
        for param in self.orig.parameters():
            param.requires_grad = False
            # Replace the last fully-connected layer
            # Parameters of newly constructed modules have requires_grad=True by default

        self.fc3 = nn.Linear(100, 2).to(device)

    def forward(self, x):
        x = self.noise(x)
        x = self.drop(x)
        x = self.orig(x)
        x = self.drop(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        x = self.noise(x)
        x = self.bn2(F.relu(self.fc(x)))
        x = self.drop(x)
        x = self.bn3(F.relu(self.fc2(x)))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        p = F.softmax(x, dim=1)
        return x, p


class InceptionV3(nn.Module):
    def __init__(self, orig_model):
        super(InceptionV3, self).__init__()
        self.num_ftrs = orig_model.fc.in_features
        self.orig = list(orig_model.children()[:-1])
        self.orig.extend([nn.Linear(self.num_ftrs, 500, bias=True)])
        self.orig.extend([nn.Linear(500, 20, bias=True)])
        self.orig.extend([nn.Linear(20, 2, bias=True)])
        self.orig = nn.Sequential(*self.orig)

    def forward(self, x):
        """
        Expects a tuple of (boundary, tumor)
        """
        x = self.orig(x)
        p = F.softmax(x, dim=1)
        return x, p


class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5).to(device)
        self.pool = nn.MaxPool2d(3, 3).to(device)
        self.pool2 = nn.AvgPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(8, 32, 5).to(device)
        self.conv3 = nn.Conv2d(32, 64, 3).to(device)
        self.conv4 = nn.Conv2d(64, 128, 3).to(device)
        self.drop = nn.Dropout(0.2).to(device)
        self.fc1 = nn.Linear(128 * 3 * 3, 512).to(device)
        self.fc2 = nn.Linear(512, 64).to(device)
        self.fc3 = nn.Linear(64, 16).to(device)
        self.fc4 = nn.Linear(16, 2).to(device)

    def forward(self, x):
        sB = x
        vals = torch.Tensor(4,x.size(0),2)
        for i in range(0, 4):
            x = sB[:, i, :, :].unsqueeze(1).to(device)
            x = x.type(torch.cuda.FloatTensor)
            x = (self.pool(self.conv1(x)))
            x = (self.pool2(self.conv2(x)))
            x = (self.pool2(self.conv3(x)))
            x = (self.pool2(self.conv4(x)))
            x = x.view(-1, 128 * 3 * 3)
            x = self.fc1(x)
            x = (self.fc2(x))
            x = (self.fc3(x))
            x = self.fc4(x)
            vals[i,:,:] = x
        # x = (vals[0,:,:]*0.3 + vals[1,:,:]*0.25 + vals[2,:,:]*0.25 + vals[3,:,:]*0.2).to(device)
        # x = vals.mean(0).to(device)
        x = vals[0,:,:].to(device)
        p = F.softmax(x, dim=1).to(device)
        return x, p


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5).to(device)
        self.pool = nn.MaxPool2d(3, 3).to(device)
        self.pool2 = nn.AvgPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(device)
        self.conv3 = nn.Conv2d(16, 64, 3).to(device)
        self.conv4 = nn.Conv2d(64, 128, 3).to(device)
        self.conv5 = nn.Conv2d(128, 200, 3).to(device)
        # self.conv6 = nn.Conv2d(200, 256, 5).to(device)
        self.fc1 = nn.Linear(200 * 4 * 1, 100).to(device)
        self.fc2 = nn.Linear(100, 64).to(device)
        # self.do = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 16).to(device)
        self.fc4 = nn.Linear(16, 2).to(device)

    def forward(self, x):
        x = x.to(device)
        x = x.type(torch.cuda.FloatTensor)
        # print(x.shape)
        x = self.pool((self.conv1(x)))
        # print(x.shape)
        x = self.pool2((self.conv2(x)))
        # print(x.shape)
        x = self.pool2((self.conv3(x)))
        # print(x.shape)
        x = self.pool2((self.conv4(x)))
        # print(x.shape)
        x = self.pool2((self.conv5(x)))
        # print(x.shape)
        # x = self.pool2((self.conv6(x)))
        x = x.view(-1, 200 * 4 * 1)
        x = (self.fc1(x))
        # print(x.shape)
        # x = self.do(x)
        x = (self.fc2(x))
        # print(x.shape)
        x = nn.Sigmoid()(self.fc3(x))
        x = self.fc4(x)
        p = F.softmax(x, dim=1).to(device)
        # print(x.shape)
        return x, p


class VGG(nn.Module):
    def __init__(self, vgg_type):
        super(VGG, self).__init__()
        orig_model = ''
        if vgg_type == 'vgg11':
            orig_model = models.vgg11(pretrained=True)
        elif vgg_type == 'vgg13':
            orig_model = models.vgg13(pretrained=True)
        elif vgg_type == 'vgg16':
            orig_model = models.vgg16(pretrained=True)
        elif vgg_type == 'vgg19':
            orig_model = models.vgg19(pretrained=True)
        elif vgg_type == 'vgg11_bn':
            orig_model = models.vgg11_bn(pretrained=True)
        elif vgg_type == 'vgg13_bn':
            orig_model = models.vgg13_bn(pretrained=True)
        elif vgg_type == 'vgg16_bn':
            orig_model = models.vgg16_bn(pretrained=True)
        elif vgg_type == 'vgg19_bn':
            orig_model = models.vgg19_bn(pretrained=True)
        self.noise = GaussianNoise()
        self.drop = nn.Dropout2d(0.6).to(device)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(100, 2).to(device)
        num_features = orig_model.classifier[0].out_features
        features = list(orig_model.classifier.children())[:-4]
        features.extend([self.noise])
        features.extend([nn.Linear(num_features, 100)])
        features.extend([self.relu])
        features.extend([self.drop])
        features.extend([self.fc])
        orig_model.classifier = nn.Sequential(*features)
        self.orig = orig_model.to(device)
        for param in self.orig.features.parameters():
            param.requires_grad = False
            # Replace the last fully-connected layer
            # Parameters of newly constructed modules have requires_grad=True by default

    def forward(self, x):
        x = self.orig(x)
        p = F.softmax(x, dim=1)
        return x, p