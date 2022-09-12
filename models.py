import torch
from torch import nn
import torchvision
from collections.abc import Iterable

class MobileNet(nn.Module):
    def __init__(self, num_classes=7, nb_layer_fix=19):
        super(MobileNet, self).__init__()
        net = torchvision.models.mobilenet_v2(pretrained=True)
        self.features = net.features[:nb_layer_fix]
        self.train_features = net.features[nb_layer_fix:]
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=256),
            nn.ReLU(True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=256, out_features=num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        #print("shape1: ", x.shape)
        x = self.train_features(x)
        #print("shape2: ", x.shape)
        #print("after feature: ", x.shape)
        x = self.avg_pooling(x)
        #print("after pooling: ", x.shape)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SqueezeNet, self).__init__()
        net = torchvision.models.squeezenet1_1(pretrained=True)
        self.features = net.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv2d(
                in_channels=512, out_channels=num_classes, 
                kernel_size=(1,1), stride=(1,1),
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
        )
    def forward(self, x):
        x = self.features(x)
        print("shape: ", x.shape)
        x = self.classifier(x)
        #print("shape after classifier: ", x.shape)
        x = x.view(x.size()[0], -1)
        #print("shape after view: ", x.shape)
        return x

class MnasNet(nn.Module):
    def __init__(self, num_classes=7):
        super(MnasNet, self).__init__()
        net = torchvision.models.mnasnet1_0(pretrained=True)
        self.features = net.layers
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        print("shape: ", x.shape)
        x = self.avg_pooling(x)
        #print("shape after features: ", x.shape)
        x = x.view(x.size()[0], -1)
        #print("shape after view: ", x.shape)
        x = self.classifier(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes=7):
        super(DenseNet, self).__init__()
        net = torchvision.models.densenet121(pretrained=True)
        self.features = net.features
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifer = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        print("shape: ", x.shape)
        x = self.avg_pooling(x)
        #print("shape after features: ", x.shape)
        x = x.view(x.size()[0], -1)
        #print("shape after view: ", x.shape)
        x = self.classifer(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet, self).__init__()
        net = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes, bias=True),
            #nn.ReLU(True),
            #nn.Linear(in_features=1000, out_features=num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        print("shape: ", x.shape)
        x = self.avg_pooling(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

# auxiliary classifiers close
class InceptionV3(nn.Module):
    def __init__(self, num_classes=7):
        super(InceptionV3, self).__init__()
        net = torchvision.models.inception_v3(pretrained=True)
        self.features = nn.Sequential(
            net.Conv2d_1a_3x3,
            net.Conv2d_2a_3x3,
            net.Conv2d_2b_3x3,
            net.maxpool1,
            net.Conv2d_3b_1x1,
            net.Conv2d_4a_3x3,
            net.maxpool2,
            net.Mixed_5b,
            net.Mixed_5c,
            net.Mixed_5d,
            net.Mixed_6a,
            net.Mixed_6b,
            net.Mixed_6c,
            net.Mixed_6d,
            net.Mixed_6e,
            net.Mixed_7a,
            net.Mixed_7b,
            net.Mixed_7c,
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        print("shape: ", x.shape)
        #print("shape after features: ", x.shape)
        x = self.avg_pooling(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x




def set_freeze_weights(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_features_weights(model, layer_names):
    set_freeze_weights(model, layer_names, True)

def unfreeze_features_weights(model, layer_names):
    set_freeze_weights(model, layer_names, False)






###############
# ELM
###############
class ELM():
    def __init__(self, input_size, h_size, num_classes, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = num_classes
        self._device = device

        self._alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._bias = torch.zeros(self._h_size, device=self._device)

        self._activation = torch.sigmoid

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def fit(self, x, t):
        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(t)


    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        return acc

#####################
# Helper Functions
#####################
def to_onehot(batch_size, num_classes, y, device):
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, num_classes).to(device)
    #y = y.type(dtype=torch.long)
    y = torch.unsqueeze(y, dim=1)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return 