import argparse
import os
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

def Adagrad_wrapper(model, step_size, num_epochs, train_loader, loss_fn, device="cpu"):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=step_size)
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for _, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
        loss_eval = 0
        # Iterate over all the dataset to compute the accumulated (summed) gradient
        with torch.no_grad():
            for _, (inputs_eval, labels_eval) in enumerate(train_loader):
                inputs_eval = inputs_eval.to(device)
                labels_eval = labels_eval.to(device)
                mini_batch_size = len(labels_eval)
                output_eval = model(inputs_eval)
                loss_eval += mini_batch_size * loss_fn(output_eval, labels_eval)
            # average the accumulated gradient, copy params to prev_params, and update params
            data_size = len(train_loader.dataset)
            loss_eval /= data_size

        print('[epoch %d] loss: %f' %
              (epoch + 1, loss_eval))


if __name__ == '__main__':
    """ train a model that will be used to compute the Inception score 

    arguments:
      data_dir (str): the data folder
      dataset_name (str): 'MNIST' or 'FashionMNIST',default: 'MNIST'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    DATA_DIR = args.data_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    
    use_gpu = True
    algorithm_to_run = 'Adagrad'
    
    # Adagrad
    batch_size = 64
    num_epochs = 60
    step_size = 1e-2
    
    # build data loaders
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    
    if dataset_name == 'MNIST':
        train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        output_model_path = os.path.join(DATA_DIR, 'resnet18_mnist_opt.pth')
    elif dataset_name == 'FashionMNIST':
        train_set = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        output_model_path = os.path.join(DATA_DIR, 'resnet18_fashion_mnist_opt.pth')
    else:
        raise Exception(
            'The "{:s}" dataset is not supported yet.'.format(dataset_name))

    train_loader = DataLoader(train_set, num_workers=2, batch_sampler=BatchSampler(
        RandomSampler(train_set, replacement=True), batch_size=batch_size, drop_last=True))
    
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    
    # try to get GPU
    device = "cpu"
    if use_gpu is True:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("no cuda device available, use cpu instead")
    
    # initialize the model
    model = resnet18(num_classes=10) # MNIST has 10 classes
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    model.to(device)
    
    loss_fn = F.cross_entropy
    
    print('training resnet18 on {:s}...'.format(dataset_name))
    Adagrad_wrapper(model, step_size, num_epochs, train_loader, loss_fn, device=device)
    
    print('Finished Training')
    
    torch.save(model.state_dict(), output_model_path)
    
    # compute training accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the %d train images: %d %%' % (
        len(train_loader.dataset), 100 * correct / total))
    
    # compute test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the %d test images: %d %%' % (
        len(test_loader.dataset), 100 * correct / total))
