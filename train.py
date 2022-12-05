import argparse
import os
import random
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from dataset import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='cifar10_train_9cnn_noise_s0.6')
parser.add_argument('-dataset_path', type=str, help='path of dataset', default='F:\Data\dataset')
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
parser.add_argument('-num_classes', type=int, help='number of classes', default=10)
parser.add_argument('-noise_rate', type=float, help='noise rate', default=0.2)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def create_dataloader():
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.RandomGrayscale(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # load dataset
    train_set = CIFAR10FromTxt(txt_path="data/cifar10_noise_s0.6.txt", root=args.dataset_path, train=True,
                               transform=transform, download=True, need_idx=False)

    # # train_set = datasets.CIFAR10(args.dataset_path, train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10(args.dataset_path, train=False, transform=test_transform, download=False)

    # # split train set into train-val set
    # # train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000])
    # # train_set, _ = torch.utils.data.random_split(train_set, [45000, 5000])
    #
    # # generate data loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    #
    # # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader, test_loader


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Set the module in training mode
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.detach().cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))  # 3*274*274->274*274*3
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            plt.close(fig)

        # print train loss and accuracy
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    # record loss and accuracy
    train_loss /= len(train_loader)  # must divide iter num
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()  # Set the module in evaluation mode
    val_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            val_loss += criterion(output, target).item()
            # val_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print val loss and accuracy
    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

    # record loss and accuracy
    val_loss_lst.append(val_loss)
    val_acc_lst.append(correct / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model, test_loader, device):
    model.eval()  # Set the module in evaluation mode
    test_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print test loss and accuracy
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    set_seed(args.seed)

    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.project_name + ' ' + now)
    os.makedirs(output_path)

    train_loader, val_loader, test_loader = create_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = CIFAR10Net().to(device)
    model = CNN9Layer(num_classes=10, input_shape=3).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    # train, validate and test
    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    # main loop
    for epoch in range(args.epochs):
        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer, epoch, device, train_loss_lst,
                                              train_acc_lst)
        val_loss_lst, val_acc_lst = validate(model, val_loader, device, val_loss_lst, val_acc_lst)

        # modify learning rate
        if epoch in [40, 80]:
            args.lr *= 0.1
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0001)

    test(model, test_loader, device)

    # plot loss and accuracy curve
    fig = plt.figure('Loss and acc', dpi=200)
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.plot(range(args.epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(args.epochs), val_acc_lst, 'b', label='val acc')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_acc.png'))
    plt.show()
    plt.close(fig)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, args.project_name + ".pth"))
