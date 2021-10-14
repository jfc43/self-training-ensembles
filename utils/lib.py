import os, csv
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import random
import torch.optim as optim

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_proxy_risk_error_detection_results(model2, dataloader_target_test, target_test_pred_labels):
    target_test_pred_labels_2, target_test_labels = get_model_pred(model2, dataloader_target_test)
    y_true = (target_test_pred_labels!=target_test_labels)
    y_pred = (target_test_pred_labels_2!=target_test_pred_labels)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return precision, recall, f1

def get_error_detection_results(record, target_test_labels, target_test_pred_labels):
    y_true = (target_test_pred_labels!=target_test_labels)
    y_pred = record
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return precision, recall, f1

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
    return acc.numpy()

def get_data(dataloader):
    X = []
    y = []
    for batch_x, batch_y in dataloader:
        X.extend(batch_x.expand(batch_x.data.shape[0], 3, 28, 28).numpy())
        y.extend(batch_y.numpy())
    return np.array(X), np.array(y)

def get_model_conf(model, dataloader):
    model.eval()
    pred_confs = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        _, batch_outputs, _ = model(batch_x)
        batch_pred_confs = np.max(batch_outputs.detach().cpu().numpy(), axis=1)
        pred_confs.extend(batch_pred_confs)
        
    return np.array(pred_confs)

def get_model_outputs(model, dataloader, return_label=False):
    model.eval()
    outputs = []
    labels = []
    for img, label in iter(dataloader):
        img, label = img.cuda(), label.cuda()
        _, class_output, _ = model(img)
        outputs.extend(class_output.cpu().detach().numpy())
        labels.extend(label.cpu().numpy())
    if return_label:
        return np.array(outputs), np.array(labels)
    else:
        return np.array(outputs)

def get_model_pred(model, dataloader):
    model.eval()
    pred_labels = []
    labels = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        _, batch_outputs, _ = model(batch_x)
        batch_pred_labels = np.argmax(batch_outputs.detach().cpu().numpy(), axis=1)
        pred_labels.extend(batch_pred_labels)
        labels.extend(batch_y)
    return np.array(pred_labels), np.array(labels)

def proxy_risk_train(model, model2, dataloader_source, dataloader_target, max_epoch, lam):
    loss_class = torch.nn.NLLLoss().cuda()
    loss_domain = torch.nn.NLLLoss().cuda()
    optimizer = optim.Adam(model2.parameters(), lr=1e-4)

    model.eval()
    model2.eval()

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    alpha = 0.1
    while i < (len_dataloader):
        # source
        s_img, s_label, domain_label = sample_batch(data_source_iter, source=True)
        if len(s_img) <= 1:
            data_source_iter = iter(dataloader_source)
            s_img, s_label, domain_label = sample_batch(data_source_iter, source=True)

        class_output, _, domain_output = model2(s_img, alpha=alpha)
        loss_s_domain = loss_domain(domain_output, domain_label)
        loss_s_label = loss_class(class_output, s_label)

        # target
        t_img, _, domain_label = sample_batch(data_target_iter, source=False)
        if len(t_img) <= 1:
            data_target_iter = iter(dataloader_target)
            t_img, _, domain_label = sample_batch(data_target_iter, source=False)

        _, output, domain_output = model2(t_img, alpha=alpha)
        loss_t_domain = loss_domain(domain_output, domain_label)

        # maximize the disagreement
        _, output_, _ = model(t_img)
        disagree = torch.norm(output - output_, p=2, dim=1).mean()
        dann_loss = loss_s_label + loss_s_domain + loss_t_domain
        # Lagrangian relaxation
        loss = - disagree + lam * dann_loss
        # if (i+1) % 10 == 0:
        #     print("disagree: {:.4f}, DANN Loss: {:.4f}".format(disagree, dann_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

def train(model2, dataloader_source, optimizer):
    model2.train()
    loss_class = torch.nn.NLLLoss().cuda()
    for s_img, s_label in iter(dataloader_source):
        s_img, s_label = s_img.cuda(), s_label.cuda()
        if s_img.shape[0] <=1: 
            continue
        class_output, _, _ = model2(s_img)
        loss = loss_class(class_output, s_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def ensemble_self_training(model2, dataloader_source, pseudo_weight, optimizer):
    model2.train()
    loss_class = torch.nn.NLLLoss(reduction="none").cuda()
    len_dataloader = len(dataloader_source)
    data_source_iter = iter(dataloader_source)

    i = 0
    while i < len_dataloader:
        # source
        s_img, s_label, s_domain_label, s_weight = sample_batch(data_source_iter, True, True)
        if len(s_img) <= 1:
            data_source_iter = iter(dataloader_source)
            s_img, s_label, s_domain_label, s_weight = sample_batch(data_source_iter, True, True)
        
        s_class_output, _, _ = model2(s_img)
        loss_s_label = loss_class(s_class_output, s_label)
     
        p_s_weight = s_weight + pseudo_weight * (s_weight == 0).type(torch.float32)
        # domain-invariant loss
        loss = (p_s_weight * loss_s_label).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

def dann_ensemble_self_training(model2, dataloader_source, dataloader_target, pseudo_weight, optimizer):
    model2.train()
    loss_class = torch.nn.NLLLoss(reduction="none").cuda()
    loss_domain = torch.nn.NLLLoss(reduction="none").cuda()
    len_dataloader = max(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    alpha = 0.1
    while i < len_dataloader:
        # source
        s_img, s_label, s_domain_label, s_weight = sample_batch(data_source_iter, True, True)
        if len(s_img) <= 1:
            data_source_iter = iter(dataloader_source)
            s_img, s_label, s_domain_label, s_weight = sample_batch(data_source_iter, True, True)
        
        s_class_output, _, s_domain_output = model2(s_img, alpha=alpha)
        loss_s_domain = loss_domain(s_domain_output, s_domain_label)
        loss_s_label = loss_class(s_class_output, s_label)

        # target
        t_img, _, t_domain_label = sample_batch(data_target_iter, False)
        if len(t_img) <= 1:
            data_target_iter = iter(dataloader_target)
            t_img, _, t_domain_label = sample_batch(data_target_iter, False)
            
        _, _, t_domain_output = model2(t_img, alpha=alpha)
        loss_t_domain = loss_domain(t_domain_output, t_domain_label)
        
        p_s_weight = s_weight + pseudo_weight * (s_weight == 0).type(torch.float32)
        # domain-invariant loss
        loss = loss_t_domain.mean() + (s_weight * loss_s_domain).mean() + (p_s_weight * loss_s_label).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

def sample_batch(data_iter, source, return_weight=False):
    try:
        if return_weight:
            img, label, weight = data_iter.next()
        else:
            img, label = data_iter.next()
    except StopIteration:
        if return_weight:
            return [], [], [], []
        else:
            return [], [], []
    # domain labels
    batch_size = len(label)
    if source:
        domain_label = torch.zeros(batch_size).long()
    else:
        domain_label = torch.ones(batch_size).long()
 
    if return_weight:
        return img.cuda(), label.cuda(), domain_label.cuda(), weight.cuda()
    else:
        return img.cuda(), label.cuda(), domain_label.cuda()

def test_disagreement(model, model2, dataloader):
    model.eval()
    model2.eval()
    n_correct, n_total = 0, 0
    for img, label in iter(dataloader):
        batch_size = len(label)
        img, label = img.cuda(), label.cuda()

        _, output_1, _ = model(img)
        _, output_2, _ = model2(img)

        pred_1 = output_1.data.max(1, keepdim=True)[1]
        pred_2 = output_2.data.max(1, keepdim=True)[1]
        n_correct += pred_1.eq(pred_2.data.view_as(pred_1)).cpu().sum()
        n_total += batch_size

    agree = n_correct.double() * 1.0 / n_total
    return 1 - agree


def test_divergence(model, dataloader, source):
    model.eval()
    loss_domain = torch.nn.NLLLoss().cuda()
    loss_list = []
    for img, label in iter(dataloader):
        batch_size = len(label)
        if source:
            domain_label = torch.zeros(batch_size).long()
        else:
            domain_label = torch.ones(batch_size).long()
        img, domain_label = img.cuda(), domain_label.cuda()

        _, _, domain_output = model(img)
        loss = loss_domain(domain_output, domain_label)
        loss_list.append(loss.detach().cpu().mean().numpy())

    return np.mean(loss_list)

def get_ensemble_model_confidence(ensemble_model, dataloader):
    ensemble_outputs = []
    N = len(ensemble_model)
    for model in ensemble_model:
        model.eval()
        outputs = []
        for img, label in iter(dataloader):
            img, label = img.cuda(), label.cuda()
            _, class_output, _ = model(img)
            outputs.extend(class_output.cpu().detach().numpy())
        ensemble_outputs.append(np.array(outputs))
    ensemble_outputs = np.array(ensemble_outputs)
    ensemble_outputs = np.swapaxes(ensemble_outputs, 0, 1)
    ensemble_confs = np.max(np.mean(ensemble_outputs, axis=1), axis=1)
    
    return ensemble_confs

def get_model_logit_outputs(model, dataloader, return_label=False):
    model.eval()
    outputs = []
    labels = []
    for img, label in iter(dataloader):
        img, label = img.cuda(), label.cuda()
        class_output = model.get_logit_output(img)
        outputs.extend(class_output.cpu().detach().numpy())
        labels.extend(label.cpu().numpy())
    if return_label:
        return np.array(outputs), np.array(labels)
    else:
        return np.array(outputs)