import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from utils.data_util import *
from utils.lib import *
from models.dann_model import DANNModel


def sample_batch(data_iter, source):
    try:
        img, label = data_iter.next()
    except StopIteration:
        return [], [], []

    # domain labels
    batch_size = len(label)
    if source:
        domain_label = torch.zeros(batch_size).long()
    else:
        domain_label = torch.ones(batch_size).long()

    return img.cuda(), label.cuda(), domain_label.cuda()

def test(model, dataloader):
    model.eval()
    n_correct, n_total = 0, 0
    for img, label in iter(dataloader):
        batch_size = len(label)
        img, label = img.cuda(), label.cuda()

        class_output, _, _ = model(img)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    acc = n_correct.double() / n_total
    return acc

def train(model, dataloader_source, dataloader_target, epoch, nepoch):
    model.train()

    loss_class = torch.nn.NLLLoss().cuda()
    loss_domain = torch.nn.NLLLoss().cuda()

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:
        # progressive
        p = float(i + epoch * len_dataloader) / nepoch / len_dataloader
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.1

        # source
        s_img, s_label, domain_label = sample_batch(data_source_iter, True)
        if len(s_img) == 0:
            data_source_iter = iter(dataloader_source)
            s_img, s_label, domain_label = sample_batch(data_source_iter, True)

        class_output, _, domain_output = model(s_img, alpha=alpha)
        loss_s_domain = loss_domain(domain_output, domain_label)
        loss_s_label = loss_class(class_output, s_label)

        # target
        t_img, _, domain_label = sample_batch(data_target_iter, False)
        if len(t_img) == 0:
            data_target_iter = iter(dataloader_target)
            t_img, _, domain_label = sample_batch(data_target_iter, False)
            
        _, _, domain_output = model(t_img, alpha=alpha)
        loss_t_domain = loss_domain(domain_output, domain_label)

        # domain-invariant loss
        loss = loss_t_domain + loss_s_domain + loss_s_label
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain domain-invariant check model')
    parser.add_argument('--source-dataset', default='mnist', choices=['mnist', 'mnist-m', 'svhn', 'usps'], type=str, help='source dataset')
    parser.add_argument('--target-dataset', default='mnist-m', choices=['mnist', 'mnist-m', 'svhn', 'usps'], type=str, help='target dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs for training')
    parser.add_argument('--test-time', action='store_true', help='test time adaptation')
    parser.add_argument('--base-dir', default='./checkpoints/dann_models/', type=str, help='dir to save model')
    parser.add_argument('--seed', type=int, default=0)
    
    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    batch_size, nepoch = args.batch_size, args.nepoch
    source_dataset = args.source_dataset
    target_dataset = args.target_dataset
    test_time = args.test_time

    assert source_dataset != target_dataset, "source and target datasets should be different!"
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    
    save_dir = os.path.join(args.base_dir, "{}_{}_{}".format(source_dataset, target_dataset, "test_time" if test_time else "training_time"))
    img_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])

    if source_dataset == "mnist":
        normalizer = transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3015, 0.3015, 0.3015))
        dataset_source = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform, download=True)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform, download=True)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif source_dataset == "mnist-m":
        normalizer = transforms.Normalize(mean=(0.4582, 0.4623, 0.4085), std=(0.1722, 0.1603, 0.1787))
        train_list = os.path.join('dataset/mnist_m/mnist_m_train_labels.txt')
        test_list = os.path.join('dataset/mnist_m/mnist_m_test_labels.txt')
        dataset_source = MNISTM(data_root='dataset/mnist_m/mnist_m_train', data_list=train_list, transform=img_transform)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test =  MNISTM(data_root='dataset/mnist_m/mnist_m_test', data_list=test_list, transform=img_transform)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle= False, num_workers=2)
    elif source_dataset == "svhn":
        normalizer = transforms.Normalize(mean=(0.4379, 0.4440, 0.4731), std=(0.1161, 0.1192, 0.1017))
        dataset_source = SVHN('dataset/svhn/', split='train', transform=img_transform, download=False)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test = SVHN('dataset/svhn/', split='test', transform=img_transform, download=False)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif source_dataset == "usps":
        normalizer = transforms.Normalize(mean=(0.2542, 0.2542, 0.2542), std=(0.3356, 0.3356, 0.3356))
        dataset_source = USPS(split="train", transform=img_transform)
        dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_source_test =  USPS(split="test", transform=img_transform)
        dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle= False, num_workers=2)

    if target_dataset == "mnist":
        dataset_target = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform, download=True)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform, download=True)
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif target_dataset == "mnist-m":
        train_list = os.path.join('dataset/mnist_m/mnist_m_train_labels.txt')
        test_list = os.path.join('dataset/mnist_m/mnist_m_test_labels.txt')
        dataset_target = MNISTM(data_root='dataset/mnist_m/mnist_m_train', data_list=train_list, transform=img_transform)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test =  MNISTM(data_root='dataset/mnist_m/mnist_m_test', data_list=test_list, transform=img_transform)
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle= False, num_workers=2)
    elif target_dataset == "svhn":
        dataset_target = SVHN('dataset/svhn/', split='train', transform=img_transform, download=False)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test = SVHN('dataset/svhn/', split='test', transform=img_transform, download=False)
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=False, num_workers=2)
    elif target_dataset == "usps":
        dataset_target = USPS(split="train", transform=img_transform)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
        dataset_target_test =  USPS(split="test", transform=img_transform)
        dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle= False, num_workers=2)
    
    if test_time:
        dataloader_target = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model Setup
    model = DANNModel(normalizer=normalizer).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(nepoch):
        train(model, dataloader_source, dataloader_target, epoch, nepoch)
        acc_s = test(model, dataloader_source_test)
        acc_t = test(model, dataloader_target_test)
        print('EPOCH {} Acc: {} {:.2f}% {} {:.2f}%'.format(epoch, source_dataset, acc_s*100, target_dataset, acc_t*100))

    torch.save(model, os.path.join(save_dir, 'checkpoint.pth'))