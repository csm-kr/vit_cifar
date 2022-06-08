import torch
import visdom
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from model_for_cifar import VisionTransformer
import torch.optim as optim
import time
import torchvision.transforms as tfs
from label_smooth_CE import LabelSmoothingCrossEntropyLoss
from auto_augment import CIFAR10Policy
import warmup_scheduler


def main():
    # 1. argparser
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=200)
    parer.add_argument('--batch_size', type=int, default=128)
    parer.add_argument('--lr', type=float, default=0.001)
    parer.add_argument('--step_size', type=int, default=100)
    parer.add_argument('--root', type=str, default='D:\data\CIFAR10')
    ops = parer.parse_args()

    # 2. device
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 3. ** visdom **
    vis = visdom.Visdom(port=2006)

    # 4. ** dataset / dataloader **
    transform_cifar = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010)),
                                        ])
    train_set = CIFAR10(root=ops.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)

    test_set = CIFAR10(root=ops.root,
                       train=False,
                       download=True,
                       transform=test_transform_cifar)

    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=ops.batch_size)

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=ops.batch_size)

    # ** 5. model **
    # num_params : 6.3 M (6304906)
    from model_new import ViT
    model = ViT(dim=384, mlp_dim=384, num_heads=12, num_layers=7,
                patch_size=8, image_size=32, is_cls_token=False,
                dropout_ratio=0.1, num_classes=10).to(device)

    # ** 6. criterion **
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropyLoss(classes=10, smoothing=0.1)

    # ** 7. optimizer **
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ops.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=5e-5)

    # ** scheduler **
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=1e-5)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer,
                                                        multiplier=1.,
                                                        total_epoch=5,
                                                        after_scheduler=base_scheduler)



    ###################################################
    #             training and pruning
    ###################################################

    print("training...")
    for epoch in range(ops.epoch):

        model.train()
        tic = time.time()
        # 11. train
        for idx, (img, target) in enumerate(train_loader):

            img = img.to(device)  # [N, 3, 28, 28]
            target = target.to(device)  # [N]
            output = model(img)  # [N, 10]
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if idx % ops.step_size == 0:
                vis.line(X=torch.ones((1, 1)) * idx + epoch * len(train_loader),
                         Y=torch.Tensor([loss]).unsqueeze(0),
                         update='append',
                         win='training_loss',
                         opts=dict(x_label='step',
                                   y_label='loss',
                                   title='loss',
                                   legend=['total_loss']))

                print('Epoch : {}\t'
                      'step : [{}/{}]\t'
                      'loss : {}\t'
                      'lr   : {}\t'
                      'time   {}\t'
                      .format(epoch,
                              idx, len(train_loader),
                              loss,
                              lr,
                              time.time() - tic))

        # test
        print('Validation of epoch [{}]'.format(epoch))
        model.eval()
        correct = 0
        val_avg_loss = 0
        total = 0
        with torch.no_grad():

            for idx, (img, target) in enumerate(test_loader):
                model.eval()
                img = img.to(device)  # [N, 3, 32, 32]
                target = target.to(device)  # [N]
                output = model(img)  # [N, 10]
                loss = criterion(output, target)

                output = torch.softmax(output, dim=1)
                # first eval
                pred, idx_ = output.max(-1)
                correct += torch.eq(target, idx_).sum().item()
                total += target.size(0)
                val_avg_loss += loss.item()

        print('Epoch {} test : '.format(epoch))
        accuracy = correct / total
        print("accuracy : {:.4f}%".format(accuracy * 100.))

        val_avg_loss = val_avg_loss / len(test_loader)
        print("avg_loss : {:.4f}".format(val_avg_loss))
        if vis is not None:
            vis.line(X=torch.ones((1, 2)) * epoch,
                     Y=torch.Tensor([accuracy, val_avg_loss]).unsqueeze(0),
                     update='append',
                     win='test_loss',
                     opts=dict(x_label='epoch',
                               y_label='test_',
                               title='test_loss',
                               legend=['accuracy', 'avg_loss']))

        scheduler.step()


if __name__ == '__main__':
    main()
