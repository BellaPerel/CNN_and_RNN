import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
# Hyper Parameters
num_epochs = 6
batch_size = 64
learning_rate = 0.002

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_flipped_and_cropped = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='./data/',
                              train=True,
                              transform=transform,
                              download=True)

test_dataset = dsets.CIFAR10(root='./data/',
                             train=False,
                             transform=transform,
                             download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

train_dataset_flip_crop = dsets.CIFAR10(root='./data/',
                                  train=True,
                                  transform=transform_flipped_and_cropped,
                                  download=True)

# Data Loader (Input Pipeline)
train_loader_flip_crop = torch.utils.data.DataLoader(dataset=train_dataset_flip_crop,
                                               batch_size=batch_size,
                                               shuffle=True)

# Get a batch of training data
images, labels = next(iter(train_loader_flip_crop))

# Print the image
plt.imshow(images[0].permute(1, 2, 0))
plt.show()


pics_flipped_cropped = []

for i, images_labels in enumerate(train_loader_flip_crop):
    pics_flipped_cropped.append(images_labels)

flip_crop_data = pics_flipped_cropped
###########################################################

transform_train_affine_jitter = transforms.Compose([
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset_4 = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_affine_jitter)

# Data Loader (Input Pipeline)
train_loader_4 = torch.utils.data.DataLoader(dataset=train_dataset_4,
                                             batch_size=batch_size,
                                             shuffle=True)

# Get a batch of training data
images, labels = next(iter(train_loader_4))

# Print the image
plt.imshow(images[0].permute(1, 2, 0))
plt.show()

pics_4 = []

for i, images_labels in enumerate(train_loader_4):
    pics_4.append(images_labels)

pics_4_data = pics_4

concatinated_train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_4])
concatinated_train_dataset = torch.utils.data.ConcatDataset([concatinated_train_dataset, train_dataset_flip_crop])

###########################################################

# Convolutional Neural Network (4 Convolutional layers)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(10),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=2),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=2)
        )
        self.fc1 = nn.Linear(64 * 3 * 3, 10)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)

        return self.logsoftmax(out)


cov_net = CNN()
if torch.cuda.is_available():
    cnn = cov_net.cuda()

loss_type = nn.NLLLoss()
optimizer = torch.optim.Adam(cov_net.parameters(), lr=learning_rate)
model_params_nums = sum(p.numel() for p in cov_net.parameters() if p.requires_grad)
print("Number of parameters: ", model_params_nums)

correct_pred = 0
total = 0
total_pred = 0
train_loss =[]
test_loss = []
train_error = []
test_accuracy = []
test_error = []
running_loss = 0.0
batch_count = 0
while 100 * correct_pred / max(total_pred, 1) <= 80:
    #print(100*correct_pred / max(total, 1))
    #100 * correct_pred / max(total_pred, 1)
    cov_net.train()
    for i, images_labels in enumerate(train_loader):
        images, labels = images_labels
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = cov_net(images)
        loss = loss_type(outputs, labels)
        loss.backward()
        optimizer.step()
        cov_net.eval()
        _, predicted = torch.max(outputs.data, 1)
        total_pred += labels.size(0)
        correct_pred += (predicted == labels).sum()
        running_loss += loss.item()
        batch_count += 1
        """
        if (i + 1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'
                  .format(num_epochs, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
        """
        cov_net.train()
    for i, images_labels in enumerate(flip_crop_data):
        images, labels = images_labels
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = cov_net(images)
        loss = loss_type(outputs, labels)
        loss.backward()
        optimizer.step()
        cov_net.eval()
        _, predicted = torch.max(outputs.data, 1)
        total_pred += labels.size(0)
        correct_pred += (predicted == labels).sum()
        running_loss += loss.item()
        batch_count += 1

        """
        if (i + 1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'
                  .format(num_epochs, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
            #print the accuracy
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_pred / total_pred))
        """
    #do the same for train_loader_4
    for i, images_labels in enumerate(pics_4_data):
        images, labels = images_labels
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = cov_net(images)
        loss = loss_type(outputs, labels)
        loss.backward()
        optimizer.step()
        cov_net.eval()
        _, predicted = torch.max(outputs.data, 1)
        total_pred += labels.size(0)
        correct_pred += (predicted == labels).sum()
        """
        if (i + 1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'
                  .format(num_epochs, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
            #print the accuracy
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_pred / total_pred))
        """
    train_loss.append(running_loss / batch_count)
    running_loss = 0.0
    batch_count = 0
    num_epochs += 1
    #train_loss.append(loss.item())
    train_error.append(100 - 100 * correct_pred / max(total_pred, 1))
    #train_loss.append(running_loss / batch_count)
    correct_pred = 0
    total_pred = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = cov_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_pred += labels.size(0)
            correct_pred += (predicted == labels).sum()
            total += labels.size(0)
            running_loss = loss_type(outputs, labels) + running_loss
            batch_count += 1
        #loss = loss_type(outputs, labels)
    test_loss.append(running_loss / batch_count)
    running_loss = 0.0
    batch_count = 0

    test_accuracy.append(100 * correct_pred / max(total_pred, 1))
    test_error.append(100 - 100 * correct_pred / max(total_pred, 1))
    print("Test Accuracy of the model on the 10000 test images: {} %".format(100 * correct_pred / max(total_pred, 1)))
    #print the accuracy
    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_pred / total_pred))

#plot error per epoch of train and test
plt.plot(train_error, label='Train Error')
plt.plot(test_error, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
#add title
plt.title('Error per Epoch')
#legend
plt.legend()
plt.show()

#plot loss per epoch of train and test
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
#add title
plt.title('Loss per Epoch')
#legend
plt.legend()
plt.show()


torch.save(cov_net.state_dict(), 'model_q1.pkl')

"""
with open("model_q1.pkl", "wb") as f:
    pickle.dump(cov_net, f)
"""