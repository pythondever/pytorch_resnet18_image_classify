import config
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import os
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset


def train(model, loss_func, dataset, optimizer, epoch, writer):
    model.train()
    batch_loss = 0
    item = 0
    for batch, (image, label) in tqdm(enumerate(dataset)):
        image = image.to(config.device)
        label = label.to(config.device)
        optimizer.zero_grad()
        output = model(image)
        _, pred = torch.max(output, 1)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        # writer.add_images("train_images", image, epoch)
        writer.add_scalar("train_loss", loss, epoch)
        print("Train Epoch = {} Loss = {}".format(epoch, loss.data.item()))
        batch_loss += loss.data.item()
        item += 1

    return batch_loss / item


def valid(model, loss_func, dataset, epoch, writer):
    model.eval()
    batch_loss = 0
    item = 0
    with torch.no_grad():
        for batch, (image, label) in tqdm(enumerate(dataset)):
            image = image.to(config.device)
            label = label.to(config.device)
            output = model(image)
            loss = loss_func(output, label)
            writer.add_images("valid_images", image, epoch)
            writer.add_scalar("valid_loss", loss, epoch)
            batch_loss += loss.data.item()
            item += 1
            print("Valid Epoch = {} Loss = {}".format(epoch, loss.data.item()))
    return batch_loss / item


def train_model(model, loss_func, optimizer, step_scheduler, num_epochs=config.epoch):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 各通道颜色的均值和方差,用于归一化
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 各通道颜色的均值和方差,用于归一化
    ])
    train_dataset = Dataset(config.train_image_path, train_transform, config.image_format)
    valid_dataset = Dataset(config.valid_image_path, valid_transform, config.image_format)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    start_epoch = 0
    # 断点继续训练
    if config.resume:
        checkpoint = torch.load(config.chkpt)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
    writer = SummaryWriter(log_dir="./runs")
    # images, _ = next(iter(train_dataloader))
    # writer.add_graph(model, images)
    for epoch in range(start_epoch + 1, num_epochs):
        train_epoch_loss = train(model, loss_func, train_dataloader, optimizer, epoch, writer)
        valid_epoch_loss = valid(model, loss_func, valid_dataloader, epoch, writer)
        step_scheduler.step()
        # 模型保存
        if epoch % config.save_model_iter == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            save_model_file = os.path.join(config.model_output_dir, "epoch_{}.pth".format(epoch))
            if not os.path.exists(config.model_output_dir):
                os.makedirs(config.model_output_dir)
            torch.save(checkpoint, save_model_file)
        if train_epoch_loss < config.best_loss or valid_epoch_loss < config.best_loss:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            save_model_file = os.path.join(config.model_output_dir, "best_{}.pth".format(epoch))
            if not os.path.exists(config.model_output_dir):
                os.makedirs(config.model_output_dir)
            torch.save(checkpoint, save_model_file)
        if epoch % 10 == 0:
            print("Epoch = {} Train Loss = {} Valid Loss = {}".format(epoch, train_epoch_loss, valid_epoch_loss))
    writer.close()


if __name__ == '__main__':
    backbone = models.resnet18(pretrained=True)
    num_fits = backbone.fc.in_features
    backbone.fc = nn.Linear(num_fits, config.num_classes)  # 替换最后一个全连接层
    model_ft = backbone.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    train_model(model_ft, criterion, optimizer_ft, scheduler, config.epoch)
