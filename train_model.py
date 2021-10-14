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
from models.model import Model
from models.dann_model import DANNModel

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

def train(model, dataloader_source):
    model.train()
    for s_img, s_label in iter(dataloader_source):
        s_img, s_label = s_img.cuda(), s_label.cuda()

        class_output, _, _ = model(s_img)
        loss_s_label = loss_class(class_output, s_label)

        loss = loss_s_label
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain model')
    parser.add_argument('--source-dataset', default='mnist', choices=['mnist', 'mnist-m', 'svhn', 'usps'], type=str, help='source dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--nepoch', default=20, type=int, help='Number of epochs for training')
    parser.add_argument('--model-type', default="typical_dnn", choices=['typical_dnn', "dann_arch"], type=str, help='given model type')
    parser.add_argument('--base-dir', default='./checkpoints/source_models/', type=str, help='dir to save model')
    parser.add_argument('--seed', type=int, default=0)
    
    # args parse
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    batch_size, nepoch = args.batch_size, args.nepoch
    source_dataset = args.source_dataset
    model_type = args.model_type
    
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    save_dir = os.path.join(args.base_dir, source_dataset)

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
    
    # Model Setup
    if model_type == "typical_dnn":
        model = Model(normalizer=normalizer).cuda()
    elif model_type == "dann_arch":
        model = DANNModel(normalizer=normalizer).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_class = torch.nn.NLLLoss().cuda()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(nepoch):
        train(model, dataloader_source)
        acc_s = test(model, dataloader_source_test)
        print('EPOCH {} Acc: {} {:.2f}%'.format(epoch, source_dataset, acc_s*100))

    torch.save(model, os.path.join(save_dir, 'checkpoint.pth'))