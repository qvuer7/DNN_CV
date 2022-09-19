from models.vgg import vgg_clasifier, vgg
from utils.utils import transform
from datasets.dataset import get_cifar10
from tqdm import tqdm
import torch

def train_one_epoch(optimizer, criterion, model, train_loader, device):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for i, (image, label) in enumerate(tqdm(train_loader)):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

    return train_loss/(i+1), 100.*correct/total



def val_one_epoch(model,criterion, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0
    for i, (image, label) in enumerate(tqdm(test_loader)):
        image = image.to(device)
        label = label.to(device)
        out = model(image)
        loss = criterion(out, label)
        test_loss+= loss.item()
        _, predicted = out.max(1)
        total+= label.size(0)
        correct += predicted.eq(label).sum().item()

    return test_loss / (i + 1), 100. * correct / total

if __name__ == '__main__':
    backbone = vgg(16)
    clasifier = vgg_clasifier(backbone = backbone, n_classes = 10)

    size = (224, 224)
    mean = (0.5,0.5,0.5)
    std = (0.5, 0.5, 0.5)
    batch_size = 1
    lr = 0.05
    transforms = transform(size = size, mean = mean , std = std)

    trainset, trainloader, testset, testloader = get_cifar10(batch_size = batch_size, transform = transforms)

    optimizer = torch.optim.SGD(clasifier.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4 )
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')
    train_one_epoch(model = clasifier, train_loader=trainloader, criterion = criterion, optimizer=optimizer,device = device)



