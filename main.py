import dataset
from torch.utils.data import DataLoader
from vit import ViT
import os
import torch
import argparse
import torch.optim as optim
from utils import Logger, AverageMeter, draw_curve
import json
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision

parser = argparse.ArgumentParser(description='ViT')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='datapath')
parser.add_argument('--data_path', default='./mnist', type=str,
                    help='datapath')
parser.add_argument('--model', default='ViT', type=str,
                    help='Deep Learning network')
parser.add_argument('--embedding_dim', default=1024, type=int,
                    help='embedding size')
parser.add_argument('--size', default=224, type=int,
                    help='img size(H==W)')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adagrad', type=str,
                    help='optimizer')
parser.add_argument('--lr', default=0.1e-2, type=float,
                    help='learning rate')
parser.add_argument('--epochs', default=300, type=int,
                    help='train epoch')
parser.add_argument('--weight_decay', default=0.000001, type=float,
                    help='weight_decay')
parser.add_argument('--gpu_id', default='2', type=str,
                    help='devices')
args = parser.parse_args()


def train(model, trn_loader, criterion, optimizer, epoch, num_epoch, train_logger):
    model.train()
    train_loss = AverageMeter()
    for i, (x,y) in enumerate(trn_loader):
        x,y = x.cuda(), y.cuda()
        y_hat = model(x)
        loss = criterion(y_hat,y)
        train_loss.update(loss.item()*10000)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(trn_loader), loss=loss*10000))
    train_logger.write([epoch, train_loss.avg])


def test(model, tst_loader, criterion, epoch, num_epoch, val_logger):
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, (x,y) in enumerate(tst_loader):
            x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            loss = criterion(y_hat,y)
            val_loss.update(loss.item()*10000)

        print("=================== TEST(Validation) Start ====================")
        print('Epoch : [{0}/{1}]  Test Loss : {loss:.4f}'.format(
                epoch, num_epoch, loss=val_loss.avg))
        print("=================== TEST(Validation) End ======================")
        val_logger.write([epoch, val_loss.avg])


def main():
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # define architecture
    if args.model == 'ViT':
        network = ViT(image_size = args.size,
                      patch_size = 32,
                      num_classes = 10,
                      dim = 1024,
                      depth = 6,
                      heads = 8,
                      mlp_dim = 2048).cuda()
    else:
        pass
    #network = nn.DataParallel(network).cuda()

    # load dataset
    torch.manual_seed(42)
    DOWNLOAD_PATH = args.data_path
    BATCH_SIZE_TRAIN = args.batch_size
    BATCH_SIZE_TEST = args.batch_size
    
    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.Resize((args.size, args.size)),
                                                      torchvision.transforms.Grayscale(3),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(DOWNLOAD_PATH, train=True, download=True, transform=transform_mnist)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_dataset = MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

    print(f"Data Loaded(trn) {len(train_dataset)}")
    print(f"Data Loaded(val) {len(val_dataset)}")

    # define criterion
    criterion = nn.CrossEntropyLoss().cuda()
    if args.optim == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.7)

    # logger
    train_logger = Logger(os.path.join(save_path, 'train_loss.log'))
    val_logger = Logger(os.path.join(save_path, 'val_loss.log'))

    # training & validation
    for epoch in range(1, args.epochs+1):
        train(network, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
        test(network, val_loader, criterion, epoch, args.epochs, val_logger)
        scheduler.step()
        if epoch%20 == 0 or epoch == args.epochs :
            torch.save(network.state_dict(), '{0}/{1}_{2}.pth'.format(save_path, args.model ,epoch))
    draw_curve(save_path, train_logger, val_logger)
    print("Process complete")

if __name__ == '__main__':
    main()