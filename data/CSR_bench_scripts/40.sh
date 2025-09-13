#!/bin/bash
# Environment Setup / Requirement / Installation
git clone https://github.com/kach/gradient-descent-the-ultimate-optimizer
cd gradient-descent-the-ultimate-optimizer
pip install torch torchvision
pip install gradient-descent-the-ultimate-optimizer
mkdir -p data

# Data / Checkpoint / Weight Download (URL)
# MNIST dataset will be automatically downloaded by torchvision

# Training
cat > mnist_gdtuo_example.py << 'EOF'
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from gradient_descent_the_ultimate_optimizer import gdtuo

class MNIST_FullyConnected(nn.Module):
    """
    A fully-connected NN for the MNIST task. This is Optimizable but not itself
    an optimizer.
    """
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_FullyConnected, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def forward(self, x):
        """Compute a prediction."""
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

BATCH_SIZE = 256
EPOCHS = 1
DEVICE = 'cpu'

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dl_train = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)

model = MNIST_FullyConnected(28 * 28, 128, 10).to(DEVICE)

optim = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
mw = gdtuo.ModuleWrapper(model, optimizer=optim)
mw.initialize()

for i in range(1, EPOCHS+1):
    running_loss = 0.0
    for j, (features_, labels_) in enumerate(dl_train):
        mw.begin()
        features, labels = torch.reshape(features_, (-1, 28 * 28)).to(DEVICE), labels_.to(DEVICE)
        pred = mw.forward(features)
        loss = F.nll_loss(pred, labels)
        mw.zero_grad()
        loss.backward(create_graph=True)
        mw.step()
        running_loss += loss.item() * features_.size(0)
        if j % 50 == 0:
            print(f"Batch {j}, Loss: {loss.item():.4f}")
    train_loss = running_loss / len(dl_train.dataset)
    print("EPOCH: {}, TRAIN LOSS: {}".format(i, train_loss))
EOF
python mnist_gdtuo_example.py

# Inference / Demonstration
cat > mnist_gdtuo_demo.py << 'EOF'
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from gradient_descent_the_ultimate_optimizer import gdtuo

class MNIST_FullyConnected(nn.Module):
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_FullyConnected, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

DEVICE = 'cpu'
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=1000, shuffle=False)

model = MNIST_FullyConnected(28 * 28, 128, 10).to(DEVICE)
optim = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
mw = gdtuo.ModuleWrapper(model, optimizer=optim)
mw.initialize()

print("Running inference demo...")
with torch.no_grad():
    for features_, labels_ in dl_test:
        features = torch.reshape(features_, (-1, 28 * 28)).to(DEVICE)
        pred = mw.forward(features)
        predicted_classes = torch.argmax(pred, dim=1)
        print(f"Sample predictions: {predicted_classes[:10].tolist()}")
        break
EOF
python mnist_gdtuo_demo.py

# Testing / Evaluation
cat > mnist_gdtuo_eval.py << 'EOF'
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from gradient_descent_the_ultimate_optimizer import gdtuo

class MNIST_FullyConnected(nn.Module):
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_FullyConnected, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

DEVICE = 'cpu'
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)

model = MNIST_FullyConnected(28 * 28, 128, 10).to(DEVICE)
optim = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
mw = gdtuo.ModuleWrapper(model, optimizer=optim)
mw.initialize()

print("Evaluating model...")
correct = 0
total = 0
with torch.no_grad():
    for features_, labels_ in dl_test:
        features, labels = torch.reshape(features_, (-1, 28 * 28)).to(DEVICE), labels_.to(DEVICE)
        pred = mw.forward(features)
        predicted = torch.argmax(pred, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
EOF
python mnist_gdtuo_eval.py