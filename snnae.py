import os
import sys
import itertools
import numpy as np
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import snntorch as snn

from matplotlib import pyplot as plt
import colorsys
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

labelfont = FontProperties()
labelfont.set_family('serif')
labelfont.set_name('Times New Roman')
labelfont.set_size(22)

axislabelfont = FontProperties()
axislabelfont.set_family('serif')
axislabelfont.set_name('Times New Roman')
axislabelfont.set_size(22)

tickfont = FontProperties()
tickfont.set_family('serif')
tickfont.set_name('Times New Roman')
tickfont.set_size(22)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams['text.usetex'] = True

# define Network
class SNNAE(nn.Module):
    
    def __init__( self, input_dim=None, beta=0.5, thresh=1.0, num_steps=5, latent_dim=5, hidden_dim=128, lr=0.001 ):
        super().__init__()

        if input_dim is None:
            print("You need to define the `input_dim` parameter")
            sys.exit()
        
        # network params
        self.input_dim = input_dim
        self.beta = beta
        self.thresh = thresh
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dtype = torch.float
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # init layers
        self.fc1_encode = nn.Linear( self.input_dim, self.hidden_dim )
        self.lif1_encode = snn.Leaky( beta=self.beta, threshold=self.thresh )
        self.fc2_encode = nn.Linear( self.hidden_dim, self.latent_dim )
        self.lif2_encode = snn.Leaky( beta=self.beta, threshold=self.thresh  )

        self.fc1_decode = nn.Linear( self.latent_dim, self.hidden_dim )
        self.lif1_decode = snn.Leaky( beta=self.beta, threshold=self.thresh  )
        self.fc2_decode = nn.Linear( self.hidden_dim, self.input_dim )
        self.lif2_decode = snn.Leaky( beta=self.beta, threshold=self.thresh  )

        # send network to device
        self.to( self.device )

        # training
        self.loss = nn.MSELoss()
        self.loss_eval = nn.MSELoss( reduction='none' )
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.loss_hist = []

    def forward(self, x):

        # init hidden states at t=0
        mem1_encode = self.lif1_encode.init_leaky()
        mem2_encode = self.lif2_encode.init_leaky()

        mem1_decode = self.lif1_decode.init_leaky()
        mem2_decode = self.lif2_decode.init_leaky()

        # record the final layers
        spk_encode_rec = []
        mem_encode_rec = []
        spk_decode_rec = []
        mem_decode_rec = []

        for step in range(self.num_steps):

            # x shape: ( N, input_dim )
            # ( N, hidden_dim )
            cur1_encode = self.fc1_encode( x )
            # ( N, hidden_dim ), ( N, hidden_dim )
            spk1_encode, mem1_encode = self.lif1_encode( cur1_encode, mem1_encode )
            # ( N, latent_dim )
            cur2_encode = self.fc2_encode( spk1_encode )
            # ( N, latent_dim ), ( N, latent_dim )
            spk2_encode, mem2_encode = self.lif2_encode( cur2_encode, mem2_encode )

            spk_encode_rec.append( spk2_encode )
            mem_encode_rec.append( mem2_encode )

        #spk_encode_rec = torch.stack( spk_encode_rec, dim=2 )
        #mem_encode_rec = torch.stack( mem_encode_rec, dim=2 )
        #print( spk_encode_rec.shape, mem_encode_rec.shape )
        
        for step in range(self.num_steps):
            # ( N, hidden_dim )
            cur1_decode = self.fc1_decode( spk_encode_rec[step] )
            # ( N, hidden_dim ), ( N, hidden_dim )
            spk1_decode, mem1_decode = self.lif1_decode( cur1_decode, mem1_decode )
            # ( N, input_dim )
            cur2_decode = self.fc2_decode( spk1_decode )
            # ( N, input_dim ), # ( N, input_dim )
            spk2_decode, mem2_decode = self.lif2_decode( cur2_decode, mem2_decode )

            spk_decode_rec.append( spk2_decode )
            mem_decode_rec.append( mem2_decode )

        return mem_decode_rec[-1]

    def train_model(self, dataloader, num_epochs=10):
        # loop epochs
        for epoch in range(num_epochs):
            train_batch = iter(dataloader)
            epoch_loss = 0
            n_events_in_epoch = 0
            for data, _ in train_batch:
                data = data.to(self.device)
                self.train()
                mem_rec = self.forward( data )
                loss_val = self.loss( mem_rec, data )
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()        
                epoch_loss += loss_val.detach().item()
                n_events_in_epoch += 1
            epoch_loss = epoch_loss / n_events_in_epoch
            self.loss_hist.append( epoch_loss )
            print( "epoch " + str(epoch) + " - " + "loss: " + str( epoch_loss ) )

    def eval_model(self, test_dataloader):
        self.eval()
        eval_losses = []
        eval_labels = []
        eval_batch = iter(test_dataloader)
        with torch.no_grad():
            for data, labels in eval_batch:
                data = data.to(self.device)
                mem_rec = self.forward( data )
                loss_vals = self.loss_eval( mem_rec, data )
                eval_losses.append( loss_vals.detach().numpy().sum(axis=-1) )
                eval_labels.append( labels.detach().numpy() )
        eval_losses = np.concatenate( eval_losses, axis=-1 ).reshape(-1)
        eval_labels = np.concatenate( eval_labels, axis=-1 ).reshape(-1)
        fpr, tpr, thresholds = roc_curve( eval_labels, eval_losses )
        roc_auc = auc(fpr, tpr)
        return eval_losses, eval_labels, roc_auc


    def plot_roc_curve(self, test_dataloader):
        eval_losses, eval_labels, roc_auc = self.eval_model(test_dataloader)
        fpr, tpr, thresholds = roc_curve(eval_labels, eval_losses)
        fpr = np.nan_to_num(fpr)
        tpr = np.nan_to_num(tpr)
        inv_fpr = 1 / fpr
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        ax.plot(tpr, inv_fpr, color='b', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [1, 0], color='gray', linestyle='--')  # Diagonal line (random guessing)
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_s$', fontsize=16)
        ax.set_ylabel(r'$\epsilon_b^{-1}$', fontsize=16)
        ax.legend(loc="upper right")
        ax.grid(True)
        plt.show()

    def plot_loss_epochs(self, log_yscale=False):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        epochs = [ i for i in range( len(self.loss_hist) ) ]
        ax.plot( epochs, self.loss_hist, color='b', lw=2, label='Training loss')
        ax.set_xlabel(r'Epoch', fontsize=16)
        ax.set_ylabel(r'Loss', fontsize=16)
        if log_yscale:
            ax.set_yscale('log')
        #ax.legend(loc="upper right")
        ax.grid(True)
        plt.show()
        