import argparse

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='cifar10_inference_score_sort')
parser.add_argument('-dataset_path', type=str, help='path of dataset', default='F:\Data\dataset')
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
parser.add_argument('-num_classes', type=int, help='number of classes', default=10)
parser.add_argument('-noise_rate', type=float, help='noise rate', default=0.2)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(args.seed)

    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = CIFAR10FromTxt("data/cifar10_noise_s0.6.txt", args.dataset_path, train=True, transform=transform,
                               download=True, need_idx=True)
    inference_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_loss = {}
    models_lst = ["output/cifar10_pretrain_9cnn_data50000 2022-12-01-19-51-42/cifar10_pretrain_9cnn_data50000.pth",
                  "output/cifar10_pretrain_9cnn_data40000 2022-12-01-21-40-33/cifar10_pretrain_9cnn_data40000.pth",
                  "output/cifar10_pretrain_9cnn_data30000 2022-12-01-23-45-37/cifar10_pretrain_9cnn_data30000.pth",
                  "output/cifar10_pretrain_9cnn_data20000 2022-12-02-11-49-29/cifar10_pretrain_9cnn_data20000.pth",
                  "output/cifar10_pretrain_9cnn_data10000 2022-12-02-11-49-44/cifar10_pretrain_9cnn_data10000.pth"]

    for i in range(len(models_lst)):
        model_path = models_lst[i]
        model = CNN9Layer(num_classes=10, input_shape=3).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels, idxs) in enumerate(inference_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # inputs = torch.unsqueeze(inputs, 1)
                outputs = model(inputs)

                criterion = nn.CrossEntropyLoss(reduction="none")
                losses = criterion(outputs, labels)

                losses = losses.detach().cpu().numpy().tolist()
                labels = labels.detach().cpu().numpy().tolist()
                idxs = idxs.detach().cpu().numpy().tolist()

                for j in range(len(idxs)):
                    raw_idx = idxs[j]
                    loss = losses[j]
                    label = labels[j]
                    if raw_idx not in sample_loss:
                        sample_loss[raw_idx] = [loss, label]
                    else:
                        x = sample_loss[raw_idx]
                        x[0] = x[0] + loss
                        sample_loss[raw_idx] = x
        print(i)

    sample_loss = sorted(sample_loss.items(), key=lambda x: x[1][0], reverse=False)

    with open("data/cifar10_noise_s0.6_sort.txt", 'w') as f:
        for sample in sample_loss:
            raw_idx = sample[0]
            loss = sample[1][0]
            label = sample[1][1]
            f.write('raw_idx:' + str(raw_idx) + ' label:' + str(label) + ' loss:' + str(loss) + '\n')
