import gc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from model.resnet50model import ResNet50
from utils.modelbuilder import ModelBuilder
from utils.utils import MetricCollector, plot_summary, mem_start, mem_end
plt.style.use('default')

# Отключаем средства дебаггинга
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

PATH = 'path_to_folder/data'
BATCH_SIZE = 512
LR = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 5 #maximum iterations before stopping train layer
USE_AMP = True
BLOCKS = 17

loss_fn = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

def get_device(benchmark=True):
    # Определим устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda' and benchmark:
        torch.backends.cudnn.benchmark = True
    print("Using {} device".format(device))
    return device

transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

transform_test = transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

model = ResNet50()

# Создадим словарь из блоков обучаемой модели
AllBlocks = {}
AllBlocks[0] = nn.Sequential(model.conv1, model.bn1, nn.ReLU())
for i in range(3): AllBlocks[1 + i] = model.layer1[i] 
for i in range(4): AllBlocks[4 + i] = model.layer2[i] 
for i in range(6): AllBlocks[8 + i] = model.layer3[i]
for i in range(3): AllBlocks[14+ i] = model.layer4[i]

def get_dataloaders():
    cifar_train = torchvision.datasets.CIFAR10(root=PATH, 
                                            train=True, 
                                            download=True, 
                                            transform=transform_train)

    # test cifar
    cifar_test = torchvision.datasets.CIFAR10(root=PATH, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)

    train_loader = DataLoader(cifar_train, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True)

    val_loader = DataLoader(cifar_test, 
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True) 
    
    return train_loader, val_loader

def train(n, epoch, model_tmp, optimizer, train_loader, device, collector):
    """Функция обучения.

    Args:
        n (int): номер блока
        epoch (int): эпоха
        model_tmp (Module): текущая обучаемая модель
        optimizer (Optimizer): оптимизатор
        train_loader (Dataloader): загрузчик данных
        device (str): устройство исполнения
        collector (MetricCollector): сборщик метрик
    """
    train_err = 0
    acc_train = 0 
    num_correct_train = 0
    num_examples_train = 0
    model.train()
    for batch in train_loader:
        X, y = batch
        X, y = X.to(device), y.to(device)

        for i in range(n): 
            X = AllBlocks[i](X)
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = model_tmp(X)
            loss = loss_fn(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Посчитаем ошибку
        train_err += loss.item() * X.size(0)
        
        # Запишем ошибку
        collector.train_batch_error.append(loss.item())

        # Посчитаем точность
        correct = torch.sum(torch.eq(torch.max(output, dim=1)[1], y)).item()
        num_examples = X.shape[0]
        # Запишем точность
        collector.train_batch_acc.append(correct / num_examples)
        
        num_correct_train += correct
        num_examples_train += num_examples
        
        # Сброс градиентов
        for param in model_tmp.parameters():
            param.grad = None
        
    acc_train = num_correct_train / num_examples_train
    train_err /= len(train_loader.dataset)
    
    # Добавим полученные значения в коллектор
    collector.train_acc[n][epoch] = acc_train
    collector.train_err[n][epoch] = train_err

def eval(n, epoch, model_tmp, val_loader, device, collector):
    """Функция инференса.

    Args:
        n (int): номер блока
        epoch (int): эпоха
        model_tmp (Module): текущая обучаемая модель
        val_loader (Dataloader): загрузчик данных
        device (str): устройство исполнения
        collector (MetricCollector): сборщик метрик
    """
    model.eval()
    valid_err = 0
    acc_val = 0
    num_correct_val = 0
    num_examples_val = 0
    with torch.no_grad():
        for batch in val_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)

            for i in range(n): 
                X = AllBlocks[i](X)
            
            output = model_tmp(X)
            loss = loss_fn(output, y)
            # Посчитаем ошибку
        
            valid_err += loss.item() * X.size(0)
            collector.val_batch_error.append(loss.item())
        
            # Посчитаем точность
            correct = torch.sum(torch.eq(torch.max(output, dim=1)[1], y)).item()
            num_examples = X.shape[0]
            collector.val_batch_acc.append(correct / num_examples)
            
            num_correct_val += correct
            num_examples_val += num_examples
    
    acc_val = num_correct_val / num_examples_val
    valid_err /= len(val_loader.dataset)
    
    # Добавим полученные значения в коллектор    
    collector.val_acc[n][epoch] = acc_val
    collector.val_err[n][epoch] = valid_err 

def main(model):
    device = get_device()
    model = model.to(device)
    train_loader, val_loader = get_dataloaders()
    
    for n in range(BLOCKS):
        ModelTmp = ModelBuilder(model, AllBlocks, n)
        ModelTmp = ModelTmp.to(device)
        optimizer = torch.optim.AdamW(ModelTmp.parameters())
        
        for epoch in range(EPOCHS):
            train(n, epoch, ModelTmp, optimizer, train_loader, device, collector)
            eval(n, epoch, ModelTmp, val_loader, device, collector)    
        for param in ModelTmp.parameters():
            param.grad = None
        ModelTmp.head = None
        
        gc.collect()
        torch.cuda.empty_cache()
        # Выведем итоговые результаты для блока

        train_acc, train_err, val_acc, val_err = collector.summary_block(n)     
        print(f"Block: {n}, Train Error: {train_err:0.3f}, Train Acc: {train_acc:0.3f},"
            f"Valid Error: {val_err:0.3f}, Valid Acc: {val_acc:0.3f}")


if __name__ == '__main__':
    collector = MetricCollector(BLOCKS, EPOCHS)
    mem_start()
    main(model, collector)
    mem_end()
    torch.save(model.state_dict(), 'model_name')
    plot_summary(collector)
