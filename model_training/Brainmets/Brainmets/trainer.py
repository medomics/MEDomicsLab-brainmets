import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path
import os
import gc
import functools 
from shutil import copyfile
from tqdm import tqdm
from apex import amp
import dill

import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from torchvision import models
from torchsummary import summary

from Brainmets.losses import *
from Brainmets.evaluation import *
from Brainmets.model_our_arch import *

class Trainer():
    def __init__(
            self,
            name,
            model,
            train_set,
            valid_set,
            test_set,
            bs,
            lr,
            max_lr,
            loss_func,
            div_factor,
            final_div_factor,
            device,
            pretrained=None):
        self.device = device
        self.name = name
        self.lr = lr
        self.bs = bs
        self.loss_function = loss_func
        self.metrics = compute_per_channel_dice
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=bs,
            shuffle=True,
            pin_memory=False)
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=bs,
            shuffle=False,
            pin_memory=False,
            num_workers=4)
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=bs,
            shuffle=False,
            pin_memory=False,
            num_workers=4)
        if model == 'ResidualUNet3D':
            self.model = ResidualUNet3D(1, 1, True).to(self.device)
            if pretrained is not None:
                print('\n\n\n')
                print(pretrained)
                state_dict = torch.load(pretrained)
                self.model.load_state_dict(state_dict)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.tmp_optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O2')
        self.max_lr = max_lr
        self.lrs = []
        self.model_state_dicts = []
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

    def fit(self, epochs, print_each_img, use_cycle=False):
        torch.cuda.empty_cache()
        self.train_losses = []
        self.valid_losses = []
        self.train_scores = []
        self.valid_scores = []
        best_val_score = -1

        self.scheduler = OneCycleLR(
            self.tmp_optimizer,
            self.max_lr,
            epochs=epochs,
            steps_per_epoch=1,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor)
        print('div_factor ', self.div_factor)
        print('final_div_factor ', self.final_div_factor)
        for epoch in range(epochs):
            self.scheduler.step()
            lr = self.tmp_optimizer.param_groups[0]['lr']
            self.lrs.append(lr)
        del self.tmp_optimizer, self.scheduler
        gc.collect()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_score = 0
            print('epoch: ' + str(epoch))
            if use_cycle:
                lr = self.lrs[epoch]
                self.optimizer.param_groups[0]['lr'] = lr
            else:
                lr = self.lr
            print(lr)
            for index, batch in tqdm(
                enumerate(
                    self.train_loader), total=len(
                    self.train_loader)):
                sample_img, sample_mask = batch
                sample_img = sample_img.to(self.device).float()
                sample_mask = sample_mask.to(self.device).float()
                predicted_mask = self.model(sample_img)
                loss = self.loss_function(predicted_mask, sample_mask)
#                 score = self.metrics(predicted_mask,sample_mask)

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
#                 loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
#                 total_score += score.item()
                if print_each_img:
                    print('batch loss: ' + str(loss.item()))
                del batch, sample_img, sample_mask, predicted_mask, loss, scaled_loss
                gc.collect()
                torch.cuda.empty_cache()
            print('total_loss: ' + str(total_loss / len(self.train_set)))
            self.train_losses.append(total_loss / len(self.train_set))
#             self.train_scores.append(total_score/len(self.train_set))
            val_score = self.val()
            if val_score > best_val_score:
                self.check_point = self
                best_val_score = val_score
                best_epoch = epoch
        self.check_point.save_checkpoint(self.check_point.name, best_epoch, best_val_score)
        self.save_checkpoint(self.name, epoch, best_val_score)
    def val(self):
        torch.cuda.empty_cache()
        self.model.eval()
        total_val_loss = 0
        total_val_score = 0
        for index, val_batch in tqdm(
            enumerate(
                self.valid_loader), total=len(
                self.valid_loader)):
            val_sample_img, val_sample_mask = val_batch
            val_sample_img = val_sample_img.to(self.device).float()
            val_sample_mask = val_sample_mask.to(self.device).float()
            del val_batch
            gc.collect()
            with torch.no_grad():
                val_predicted_mask = self.model(val_sample_img)
            val_loss = self.loss_function(val_predicted_mask, val_sample_mask)
            val_score = self.metrics(val_predicted_mask, val_sample_mask)
            total_val_loss += val_loss.item()
            total_val_score += val_score.item()
            del val_sample_img, val_sample_mask, val_predicted_mask, val_loss, val_score
            gc.collect()
        print('total_valid_score: ' + str(total_val_score / len(self.valid_set)))
        torch.cuda.empty_cache()
        self.valid_losses.append(total_val_loss / len(self.valid_set))
        self.valid_scores.append(total_val_score / len(self.valid_set))
        return total_val_score / len(self.valid_set)

    def predict(self):
        self.model.eval()
        total_test_loss = 0
        total_test_score = 0
        for index, test_batch in tqdm(
            enumerate(
                self.test_loader), total=len(
                self.test_loader)):
            test_sample_img, test_sample_mask = test_batch
            test_sample_img = test_sample_img.to(self.device).float()
            test_sample_mask = test_sample_mask.to(self.device).float()
            del test_batch
            gc.collect()
            with torch.no_grad():
                test_predicted_mask = self.model(test_sample_img)
            test_loss = self.loss_function(
                test_predicted_mask, test_sample_mask)
            test_score = self.metrics(test_predicted_mask, test_sample_mask)
            total_test_loss += test_loss.item()
            total_test_score += test_score.item()
            del test_sample_img, test_sample_mask, test_predicted_mask, test_loss, test_score
            gc.collect()
        print('test_score: ' + str(total_test_score / len(self.test_set)))
        torch.cuda.empty_cache()
        self.test_score = total_test_score / len(self.test_set)
        return total_test_score / len(self.test_set)

    def save_checkpoint(self, name, epoch, val_score):
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if not os.path.exists('./results/' + name):
            os.mkdir('./results/' + name)
        torch.save(self.model.state_dict(), './results/' + name + '/epoch_' + str(epoch) + '_val_score=' + str(val_score) + '.pth')


    @staticmethod
    def load_best_checkpoint(name):
        checkpoints = sorted([checkpoint for checkpoint in os.listdir(
            './results/' + name) if checkpoint.startswith('epoch')])
        best_epoch = np.argmax([float(checkpoint.split('=')[1].split('.')[
                               1][:10]) for checkpoint in checkpoints])
        best_epoch = int(checkpoints[best_epoch].split('_')[1])
        print('best_epoch: ', best_epoch)
        best_checkpoint = [
            checkpoint for checkpoint in checkpoints if checkpoint.startswith(
                'epoch_' + str(best_epoch))][0]
        print([
            checkpoint for checkpoint in checkpoints if checkpoint.startswith(
                'epoch_' + str(best_epoch))])
#         return torch.load('./results/' + name + '/' + best_checkpoint)
        return torch.load(
            open(
                './results/' +
                name +
                '/' +
                best_checkpoint,
                'rb'))