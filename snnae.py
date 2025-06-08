import os
import sys
import itertools
import numpy as np
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import nir
import snntorch as snn
from snntorch.export_nir import export_to_nir

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
    
    def __init__( self, input_dim=None, beta=0.5, thresh=1.0, num_steps=5, layers=[128,64], latent_dim=5, hidden_dim=128, lr=0.001, quant=False, qpr=8 ):
        super().__init__()

        if input_dim is None:
            print("You need to define the `input_dim` parameter")
            sys.exit()
        
        # network params
        self.input_dim = input_dim
        self.beta = beta
        self.thresh = thresh
        self.num_steps = num_steps
        self.encoder_layer_dims = layers.copy()
        self.decoder_layer_dims = layers.copy()
        self.decoder_layer_dims.reverse()
        self.num_layers = len(layers)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.quant = quant
        self.qpr = qpr
        self.dtype = torch.float
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        if not self.quant:    
            # init encoder layers
            self.encoder_layers = []
            self.encoder_layers.append( nn.Linear( self.input_dim, self.encoder_layer_dims[0] ) )
            self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
            if self.num_layers > 1:
                for i, _ in enumerate(self.encoder_layer_dims):
                    if i == self.num_layers-1:
                        self.encoder_layers.append( nn.Linear( self.encoder_layer_dims[i], self.latent_dim )  )
                        self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
                    else:
                        self.encoder_layers.append( nn.Linear( self.encoder_layer_dims[i], self.encoder_layer_dims[i+1] )  )
                        self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
            else:
                self.encoder_layers.append( nn.Linear( self.encoder_layer_dims[0], self.latent_dim )  )
                self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
    
            # init decoder layers
            self.decoder_layers = []
            self.decoder_layers.append( nn.Linear( self.latent_dim, self.decoder_layer_dims[0] ) )
            self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
            if self.num_layers > 1:
                for i, _ in enumerate(self.decoder_layer_dims):
                    if i == self.num_layers-1:
                        self.decoder_layers.append( nn.Linear( self.decoder_layer_dims[i], self.input_dim )  )
                        self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh*1000 ) )
                    else:
                        self.decoder_layers.append( nn.Linear( self.decoder_layer_dims[i], self.decoder_layer_dims[i+1] )  )
                        self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
            else:
                self.decoder_layers.append( nn.Linear( self.decoder_layer_dims[0], self.input_dim )  )
                self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh*1000 ) )
        else:
            import brevitas.nn as qnn
            from snntorch.functional import quant
            self.quant_input = qnn.QuantIdentity(bit_width=self.qpr, return_quant_tensor=True)
            q_lif = quant.state_quant(num_bits=self.qpr, uniform=True, threshold=self.thresh)
            # init encoder layers
            self.encoder_layers = []
            self.encoder_layers.append( qnn.QuantLinear(self.input_dim, self.encoder_layer_dims[0], bias=True, weight_bit_width=self.qpr) )
            self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh, state_quant=q_lif ) )
            if self.num_layers > 1:
                for i, _ in enumerate(self.encoder_layer_dims):
                    if i == self.num_layers-1:
                        self.encoder_layers.append( qnn.QuantLinear(self.encoder_layer_dims[i],self.latent_dim, bias=True, weight_bit_width=self.qpr)  )
                        self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh, state_quant=q_lif ) )
                    else:
                        self.encoder_layers.append( qnn.QuantLinear(self.encoder_layer_dims[i], self.encoder_layer_dims[i+1], bias=True, weight_bit_width=self.qpr)  )
                        self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh, state_quant=q_lif ) )
            else:
                self.encoder_layers.append( qnn.QuantLinear(self.encoder_layer_dims[0], self.latent_dim, bias=True, weight_bit_width=self.qpr)  )
                self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh, state_quant=q_lif ) )
    
            # init decoder layers
            self.decoder_layers = []
            self.decoder_layers.append( qnn.QuantLinear(self.latent_dim, self.decoder_layer_dims[0], bias=True, weight_bit_width=self.qpr) )
            self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh, state_quant=q_lif ) )
            if self.num_layers > 1:
                for i, _ in enumerate(self.decoder_layer_dims):
                    if i == self.num_layers-1:
                        self.decoder_layers.append( qnn.QuantLinear(self.decoder_layer_dims[i], self.input_dim, bias=True, weight_bit_width=self.qpr)  )
                        self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh*1000, state_quant=q_lif ) )
                    else:
                        self.decoder_layers.append( qnn.QuantLinear(self.decoder_layer_dims[i], self.decoder_layer_dims[i+1], bias=True, weight_bit_width=self.qpr)  )
                        self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh, state_quant=q_lif ) )
            else:
                self.decoder_layers.append( qnn.QuantLinear(self.decoder_layer_dims[0], self.input_dim, bias=True, weight_bit_width=self.qpr)  )
                self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh*1000, state_quant=q_lif ) )

        self.encoder_layers = nn.ModuleList( self.encoder_layers )
        self.decoder_layers = nn.ModuleList( self.decoder_layers )

        print(self.encoder_layers)
        print(self.decoder_layers)
        
        # send network to device
        self.to( self.device )

        # training
        self.loss = nn.MSELoss()
        self.loss_eval = nn.MSELoss( reduction='none' )
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.loss_hist = []
        self.spikes = []

    def forward(self, x):

        if self.quant:
            x = self.quant_input( x )
        
        self.spikes = []

        # init hidden states at t=0
        #mem1_encode = self.lif1_encode.init_leaky()
        #mem2_encode = self.lif2_encode.init_leaky()

        #mem1_decode = self.lif1_decode.init_leaky()
        #mem2_decode = self.lif2_decode.init_leaky()

        # init hidden states at t=0
        mem_encode = [ lif.init_leaky() for i,lif in enumerate(self.encoder_layers) if i%2==1 ]
        mem_decode = [ lif.init_leaky() for i,lif in enumerate(self.decoder_layers) if i%2==1 ]

        # record the final layers
        spk_encode_rec = []
        mem_encode_rec = []
        spk_decode_rec = []
        mem_decode_rec = []

        #for step in range(self.num_steps):

            # x shape: ( N, input_dim )
            # ( N, hidden_dim )
            #cur1_encode = self.fc1_encode( x )
            # ( N, hidden_dim ), ( N, hidden_dim )
            #spk1_encode, mem1_encode = self.lif1_encode( cur1_encode, mem1_encode )
            # ( N, latent_dim )
            #cur2_encode = self.fc2_encode( spk1_encode )
            # ( N, latent_dim ), ( N, latent_dim )
            #spk2_encode, mem2_encode = self.lif2_encode( cur2_encode, mem2_encode )

            #spk_encode_rec.append( spk2_encode )
            #mem_encode_rec.append( mem2_encode )
        
        #for step in range(self.num_steps):
            # ( N, hidden_dim )
            #cur1_decode = self.fc1_decode( spk_encode_rec[step] )
            # ( N, hidden_dim ), ( N, hidden_dim )
            #spk1_decode, mem1_decode = self.lif1_decode( cur1_decode, mem1_decode )
            # ( N, input_dim )
            #cur2_decode = self.fc2_decode( spk1_decode )
            # ( N, input_dim ), # ( N, input_dim )
            #spk2_decode, mem2_decode = self.lif2_decode( cur2_decode, mem2_decode )

            #spk_decode_rec.append( spk2_decode )
            #mem_decode_rec.append( mem2_decode )


        for step in range(self.num_steps):
            cur_encode = self.encoder_layers[0]( x )
            spk_encode, mem_encode[0] = self.encoder_layers[1]( cur_encode, mem_encode[0] )
            self.spikes += spk_encode
            if self.num_layers > 1:
                for i in range(self.num_layers+1):
                    if i==0:
                        continue
                    cur_encode = self.encoder_layers[2*i]( spk_encode )
                    spk_encode, mem_encode[i] = self.encoder_layers[2*i+1]( cur_encode, mem_encode[i] )
                    self.spikes += spk_encode
            else:
                cur_encode = self.encoder_layers[2]( spk_encode )
                spk_encode, mem_encode[1] = self.encoder_layers[3]( cur_encode, mem_encode[1] )
                self.spikes += spk_encode
            spk_encode_rec.append( spk_encode )
            mem_encode_rec.append( mem_encode[-1] )

        for step in range(self.num_steps):
            cur_decode = self.decoder_layers[0]( spk_encode_rec[step] )
            spk_decode, mem_decode[0] = self.decoder_layers[1]( cur_decode, mem_decode[0] )
            self.spikes += spk_decode
            if self.num_layers > 1:
                for i in range(self.num_layers+1):
                    if i==0:
                        continue
                    cur_decode = self.decoder_layers[2*i]( spk_decode )
                    spk_decode, mem_decode[i] = self.decoder_layers[2*i+1]( cur_decode, mem_decode[i] )
                    self.spikes += spk_decode
            else:
                cur_decode = self.decoder_layers[2]( spk_decode )
                spk_decode, mem_decode[1] = self.decoder_layers[3]( cur_decode, mem_decode[1] )
                self.spikes += spk_decode
            spk_decode_rec.append( spk_decode )
            mem_decode_rec.append( mem_decode[-1] )

        return mem_decode_rec[-1]

    def encode(self, x):
        self.eval()
        if self.quant:
            x = self.quant_input( x )
        
        latent_spikes = []

        # init hidden states at t=0
        mem_encode = [ lif.init_leaky() for i,lif in enumerate(self.encoder_layers) if i%2==1 ]

        # record the final layers
        spk_encode_rec = []
        mem_encode_rec = []

        for step in range(self.num_steps):
            cur_encode = self.encoder_layers[0]( x )
            spk_encode, mem_encode[0] = self.encoder_layers[1]( cur_encode, mem_encode[0] )
            if self.num_layers > 1:
                for i in range(self.num_layers+1):
                    if i==0:
                        continue
                    cur_encode = self.encoder_layers[2*i]( spk_encode )
                    spk_encode, mem_encode[i] = self.encoder_layers[2*i+1]( cur_encode, mem_encode[i] )
            else:
                cur_encode = self.encoder_layers[2]( spk_encode )
                spk_encode, mem_encode[1] = self.encoder_layers[3]( cur_encode, mem_encode[1] )
            latent_spikes.append( spk_encode )
        self.train()
        return latent_spikes

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
        inv_fpr = 1 / fpr
        target_tpr = 0.3
        index = np.argmin(np.abs(tpr - target_tpr))
        inv_fpr_at_0_3 = inv_fpr[index]
        return tpr, inv_fpr, roc_auc, inv_fpr_at_0_3

    def eval_sparsity(self, test_dataloader):
        self.eval()
        eval_batch = iter(test_dataloader)
        eval_spikes = []
        with torch.no_grad():
            for data, labels in eval_batch:
                data = data.to(self.device)
                mem_rec = self.forward( data )
                eval_spikes += np.mean( [x for xs in self.spikes for x in xs] )
        return np.mean(eval_spikes)

    def plot_roc_curve(self, test_dataloader, labels=None, savefig=None, show_plots=True):
        eval_losses, eval_labels, roc_auc, inv_fpr_at_0_3 = self.eval_model(test_dataloader)
        fpr, tpr, thresholds = roc_curve(eval_labels, eval_losses)
        fpr = np.nan_to_num(fpr)
        tpr = np.nan_to_num(tpr)
        inv_fpr = 1 / fpr
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        if labels:
            ax.plot(tpr, inv_fpr, color='b', lw=2, label=f'{labels[0]}, AUC={roc_auc:.3f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        else:
            ax.plot(tpr, inv_fpr, color='b', lw=2, label=f'AUC={roc_auc:.2f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        ax.plot([1-i/10000 for i in range(10000)], [1/(1-i/10000) for i in range(10000)], color='gray', linestyle='--')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_s$', fontsize=16)
        ax.set_ylabel(r'$\epsilon_b^{-1}$', fontsize=16)
        ax.set_ylim(1,1000)
        ax.legend(loc="upper right")
        ax.grid(True)
        plt.tight_layout()
        if savefig:
            fig.savefig(savefig, dpi=500)
        if show_plots:
            plt.show()

    def plot_roc_curves(self, test_dataloaders, labels, colors=['b','r','g','m'], savefig=None, show_plots=True):
        plt.clf()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        aucs = []
        for i, test_dataloader in enumerate(test_dataloaders):
            eval_losses, eval_labels, roc_auc, inv_fpr_at_0_3 = self.eval_model(test_dataloader)
            aucs.append(roc_auc)
            fpr, tpr, thresholds = roc_curve(eval_labels, eval_losses)
            fpr = np.nan_to_num(fpr)
            tpr = np.nan_to_num(tpr)
            inv_fpr = 1 / fpr
            if labels:
                ax.plot(tpr, inv_fpr, color=colors[i], lw=2, label=f'{labels[i]}, AUC={roc_auc:.3f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
            else:
                ax.plot(tpr, inv_fpr, color=colors[i], lw=2, label=f'AUC={roc_auc:.2f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        ax.plot([1-i/10000 for i in range(10000)], [1/(1-i/10000) for i in range(10000)], color='gray', linestyle='--')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_s$', fontsize=16)
        ax.set_ylabel(r'$\epsilon_b^{-1}$', fontsize=16)
        ax.set_ylim(1,1000)
        ax.legend(loc="upper right")
        ax.grid(True)
        for i, label in enumerate(labels):
            print( label+": "+str(aucs[i]) )
        plt.tight_layout()
        if savefig:
            fig.savefig(savefig, dpi=500)
        if show_plots:
            plt.show()

    def plot_latent_space_reps(self, test_dataloaders, labels, xlabels=None, ylims=None, savefig=None, show_plots=True):
        if xlabels == None:
            xlabels = [i for i in range(self.num_steps*self.latent_dim)]
        with torch.no_grad():
            for j, test_dataloader in enumerate(test_dataloaders):
                plt.clf()
                train_batch = iter(test_dataloader)
                latent_spikes = []
                nn = 0
                for data, _ in train_batch:
                    nn += len(data)
                    latent_spikes.append( self.encode( data ) )
                latent_spikes = np.array(latent_spikes)
                ld = np.reshape( latent_spikes, [nn,self.num_steps,self.latent_dim] )
                ld = np.reshape( np.transpose(ld.mean(axis=0)), [self.num_steps * self.latent_dim] )
                plt.bar([i for i in range(self.num_steps*self.latent_dim)], ld, facecolor='none', edgecolor='black', linewidth=2, label=labels[j])
                plt.xticks([i for i in range(self.num_steps*self.latent_dim)], xlabels, fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('latent dim at step $t$', fontsize=16)
                plt.ylabel('spiking activity', fontsize=16)
                plt.legend( loc='best', fontsize=16 )
                if ylims != None:
                    plt.ylim(ylims)
                #plt.tight_layout()
                if savefig:
                    plt.savefig(savefig, dpi=500, bbox_inches="tight")
                if show_plots:
                    plt.show()

    def plot_loss_epochs(self, log_yscale=False, savefig=None, show_plots=True):
        plt.clf()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        epochs = [ i for i in range( len(self.loss_hist) ) ]
        ax.plot( epochs, self.loss_hist, color='black', lw=2, label='SNN-AE')
        ax.set_xlabel(r'Epoch', fontsize=16)
        ax.set_ylabel(r'Loss', fontsize=16)
        if log_yscale:
            ax.set_yscale('log')
        ax.legend(loc="best", fontsize=16)
        ax.tick_params(labelsize=14)
        #plt.tight_layout()
        ax.grid(True)
        if savefig:
            fig.savefig(savefig, dpi=500, bbox_inches="tight")
        if show_plots:
            plt.show()



# define Network
class SNNAEspike(nn.Module):
    
    def __init__( self, input_dim=None, beta=0.5, thresh=1.0, layers=[128,64], latent_dim=5, hidden_dim=128, lr=0.001 ):
        super().__init__()

        if input_dim is None:
            print("You need to define the `input_dim` parameter")
            sys.exit()
        
        # network params
        self.input_dim = input_dim
        self.beta = beta
        self.thresh = thresh
        self.encoder_layer_dims = layers.copy()
        self.decoder_layer_dims = layers.copy()
        self.decoder_layer_dims.reverse()
        self.num_layers = len(layers)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dtype = torch.float
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # init layers
        #self.fc1_encode = nn.Linear( self.input_dim, self.hidden_dim )
        #self.lif1_encode = snn.Leaky( beta=self.beta, threshold=self.thresh )
        #self.fc2_encode = nn.Linear( self.hidden_dim, self.latent_dim )
        #self.lif2_encode = snn.Leaky( beta=self.beta, threshold=self.thresh  )

        #self.fc1_decode = nn.Linear( self.latent_dim, self.hidden_dim )
        #self.lif1_decode = snn.Leaky( beta=self.beta, threshold=self.thresh  )
        #self.fc2_decode = nn.Linear( self.hidden_dim, self.input_dim )
        #self.lif2_decode = snn.Leaky( beta=self.beta, threshold=self.thresh  )

        # init encoder layers
        self.encoder_layers = []
        self.encoder_layers.append( nn.Linear( self.input_dim, self.encoder_layer_dims[0] ) )
        self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
        if self.num_layers > 1:
            for i, _ in enumerate(self.encoder_layer_dims):
                if i == self.num_layers-1:
                    self.encoder_layers.append( nn.Linear( self.encoder_layer_dims[i], self.latent_dim )  )
                    self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
                else:
                    self.encoder_layers.append( nn.Linear( self.encoder_layer_dims[i], self.encoder_layer_dims[i+1] )  )
                    self.encoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )

        # init decoder layers
        self.decoder_layers = []
        self.decoder_layers.append( nn.Linear( self.latent_dim, self.decoder_layer_dims[0] ) )
        self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )
        if self.num_layers > 1:
            for i, _ in enumerate(self.decoder_layer_dims):
                if i == self.num_layers-1:
                    self.decoder_layers.append( nn.Linear( self.decoder_layer_dims[i], self.input_dim )  )
                    self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh*1000 ) )
                else:
                    self.decoder_layers.append( nn.Linear( self.decoder_layer_dims[i], self.decoder_layer_dims[i+1] )  )
                    self.decoder_layers.append( snn.Leaky( beta=self.beta, threshold=self.thresh ) )

        self.encoder_layers = nn.ModuleList( self.encoder_layers )
        self.decoder_layers = nn.ModuleList( self.decoder_layers )

        print(self.encoder_layers)
        print(self.decoder_layers)
        
        # send network to device
        self.to( self.device )

        # training
        self.loss = nn.MSELoss()
        self.loss_eval = nn.MSELoss( reduction='none' )
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.loss_hist = []
        self.spike_means = []

    def forward(self, x):

        # (N, steps, n_features)
        num_steps = x.shape[0]

        self.spikes = []

        # init hidden states at t=0
        #mem1_encode = self.lif1_encode.init_leaky()
        #mem2_encode = self.lif2_encode.init_leaky()

        #mem1_decode = self.lif1_decode.init_leaky()
        #mem2_decode = self.lif2_decode.init_leaky()

        # init hidden states at t=0
        mem_encode = [ lif.init_leaky() for i,lif in enumerate(self.encoder_layers) if i%2==1 ]
        mem_decode = [ lif.init_leaky() for i,lif in enumerate(self.decoder_layers) if i%2==1 ]

        # record the final layers
        spk_encode_rec = []
        mem_encode_rec = []
        spk_decode_rec = []
        mem_decode_rec = []

        #for step in range(self.num_steps):

            # x shape: ( N, input_dim )
            # ( N, hidden_dim )
            #cur1_encode = self.fc1_encode( x )
            # ( N, hidden_dim ), ( N, hidden_dim )
            #spk1_encode, mem1_encode = self.lif1_encode( cur1_encode, mem1_encode )
            # ( N, latent_dim )
            #cur2_encode = self.fc2_encode( spk1_encode )
            # ( N, latent_dim ), ( N, latent_dim )
            #spk2_encode, mem2_encode = self.lif2_encode( cur2_encode, mem2_encode )

            #spk_encode_rec.append( spk2_encode )
            #mem_encode_rec.append( mem2_encode )
        
        #for step in range(self.num_steps):
            # ( N, hidden_dim )
            #cur1_decode = self.fc1_decode( spk_encode_rec[step] )
            # ( N, hidden_dim ), ( N, hidden_dim )
            #spk1_decode, mem1_decode = self.lif1_decode( cur1_decode, mem1_decode )
            # ( N, input_dim )
            #cur2_decode = self.fc2_decode( spk1_decode )
            # ( N, input_dim ), # ( N, input_dim )
            #spk2_decode, mem2_decode = self.lif2_decode( cur2_decode, mem2_decode )

            #spk_decode_rec.append( spk2_decode )
            #mem_decode_rec.append( mem2_decode )


        for step in range(num_steps):
            # (B, steps, layer[0])
            cur_encode = self.encoder_layers[0]( x )
            # (B, steps, layer[1])
            spk_encode, mem_encode[0] = self.encoder_layers[1]( cur_encode, mem_encode[0] )
            self.spike_means.append( torch.mean( spk_encode ) )
            if self.num_layers > 1:
                for i in range(self.num_layers+1):
                    if i==0:
                        continue
                    cur_encode = self.encoder_layers[2*i]( spk_encode )
                    # (B, steps, layer[i])
                    spk_encode, mem_encode[i] = self.encoder_layers[2*i+1]( cur_encode, mem_encode[i] )
                    self.spike_means.append( torch.mean( spk_encode ) )
                    # (B, steps, layer[i])
            spk_encode_rec.append( spk_encode )
            #mem_encode_rec.append( mem_encode[-1] )

        for step in range(num_steps):
            cur_decode = self.decoder_layers[0]( spk_encode_rec[step] )
            spk_decode, mem_decode[0] = self.decoder_layers[1]( cur_decode, mem_decode[0] )
            self.spike_means.append( torch.mean( spk_decode ) )
            if self.num_layers > 1:
                for i in range(self.num_layers+1):
                    if i==0:
                        continue
                    cur_decode = self.decoder_layers[2*i]( spk_decode )
                    spk_decode, mem_decode[i] = self.decoder_layers[2*i+1]( cur_decode, mem_decode[i] )
                    self.spike_means.append( torch.mean( spk_decode ) )
            #spk_decode_rec.append( spk_decode )

        return torch.mean(spk_decode, 1)

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
                loss_val = self.loss( mem_rec, torch.mean( data, 1 ) )
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()        
                epoch_loss += loss_val.detach().item()
                n_events_in_epoch += 1
            epoch_loss = epoch_loss / n_events_in_epoch
            self.loss_hist.append( epoch_loss )
            #print( "epoch " + str(epoch) + " - " + "loss: " + str( epoch_loss ) )

    def eval_model(self, test_dataloader):
        self.eval()
        eval_losses = []
        eval_labels = []
        eval_batch = iter(test_dataloader)
        with torch.no_grad():
            for data, labels in eval_batch:
                data = data.to(self.device)
                spk_rec = self.forward( data )
                loss_vals = self.loss_eval( spk_rec, torch.mean( data, 1 ) )
                eval_losses.append( loss_vals.detach().numpy().sum(axis=-1) )
                eval_labels.append( labels.detach().numpy() )
        eval_losses = np.concatenate( eval_losses, axis=-1 ).reshape(-1)
        eval_labels = np.concatenate( eval_labels, axis=-1 ).reshape(-1)
        fpr, tpr, thresholds = roc_curve( eval_labels, eval_losses )
        roc_auc = auc(fpr, tpr)
        inv_fpr = 1 / fpr
        target_tpr = 0.3
        index = np.argmin(np.abs(tpr - target_tpr))
        inv_fpr_at_0_3 = inv_fpr[index]
        return tpr, inv_fpr, roc_auc, inv_fpr_at_0_3

    def eval_sparsity(self, test_dataloader):
        self.eval()
        eval_batch = iter(test_dataloader)
        eval_spikes = []
        with torch.no_grad():
            for data, labels in eval_batch:
                data = data.to(self.device)
                spk_rec = self.forward( data )
                eval_spikes += self.spikes
        return np.mean(eval_spikes)

    def plot_roc_curve(self, test_dataloader, labels=None, savefig=None):
        eval_losses, eval_labels, roc_auc, inv_fpr_at_0_3 = self.eval_model(test_dataloader)
        fpr, tpr, thresholds = roc_curve(eval_labels, eval_losses)
        fpr = np.nan_to_num(fpr)
        tpr = np.nan_to_num(tpr)
        inv_fpr = 1 / fpr
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        if labels:
            ax.plot(tpr, inv_fpr, color='b', lw=2, label=f'{labels[0]}, AUC={roc_auc:.3f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        else:
            ax.plot(tpr, inv_fpr, color='b', lw=2, label=f'AUC={roc_auc:.2f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        ax.plot([1-i/10000 for i in range(10000)], [1/(1-i/10000) for i in range(10000)], color='gray', linestyle='--')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_s$', fontsize=16)
        ax.set_ylabel(r'$\epsilon_b^{-1}$', fontsize=16)
        ax.set_ylim(1,1000)
        ax.legend(loc="upper right")
        ax.grid(True)
        if savefig:
            fig.savefig(savefig, dpi=500)
        plt.show()

    def plot_roc_curves(self, test_dataloaders, labels, colors=['b','r','g','m'], savefig=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        aucs = []
        for i, test_dataloader in enumerate(test_dataloaders):
            eval_losses, eval_labels, roc_auc, inv_fpr_at_0_3 = self.eval_model(test_dataloader)
            aucs.append(roc_auc)
            fpr, tpr, thresholds = roc_curve(eval_labels, eval_losses)
            fpr = np.nan_to_num(fpr)
            tpr = np.nan_to_num(tpr)
            inv_fpr = 1 / fpr
            if labels:
                ax.plot(tpr, inv_fpr, color=colors[i], lw=2, label=f'{labels[i]}, AUC={roc_auc:.3f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
            else:
                ax.plot(tpr, inv_fpr, color=colors[i], lw=2, label=f'AUC={roc_auc:.2f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        ax.plot([1-i/10000 for i in range(10000)], [1/(1-i/10000) for i in range(10000)], color='gray', linestyle='--')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_s$', fontsize=16)
        ax.set_ylabel(r'$\epsilon_b^{-1}$', fontsize=16)
        ax.set_ylim(1,1000)
        ax.legend(loc="upper right")
        ax.grid(True)
        for i, label in enumerate(labels):
            print( label+": "+str(aucs[i]) )
        if savefig:
            fig.savefig(savefig, dpi=500)
        plt.show()

    def plot_loss_epochs(self, log_yscale=False, savefig=None):
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
        if savefig:
            fig.savefig(savefig, dpi=500)
        plt.show()





# define Network
class AE(nn.Module):
    
    def __init__( self, input_dim=None, layers=[128,64], latent_dim=5, hidden_dim=128, lr=0.001, quant=False, qpr=8 ):
        super().__init__()
        # network params
        self.input_dim = input_dim
        self.encoder_layer_dims = layers.copy()
        self.decoder_layer_dims = layers.copy()
        self.decoder_layer_dims.reverse()
        self.num_layers = len(layers)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.quant = quant
        self.qpr = qpr
        self.dtype = torch.float
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        if self.input_dim is None:
            print("You need to define the `input_dim` parameter")
            sys.exit()

        if not self.quant:
            # init encoder layers
            self.encoder_layers = []
            self.encoder_layers.append( nn.Linear( self.input_dim, self.encoder_layer_dims[0] ) )
            self.encoder_layers.append( nn.ReLU() )
            if self.num_layers > 1:
                for i, _ in enumerate(self.encoder_layer_dims):
                    if i == self.num_layers-1:
                        self.encoder_layers.append( nn.Linear( self.encoder_layer_dims[i], self.latent_dim )  )
                    else:
                        self.encoder_layers.append( nn.Linear( self.encoder_layer_dims[i], self.encoder_layer_dims[i+1] )  )
                        self.encoder_layers.append( nn.ReLU() )
        
            # init decoder layers
            self.decoder_layers = []
            self.decoder_layers.append( nn.Linear( self.latent_dim, self.decoder_layer_dims[0] ) )
            self.decoder_layers.append( nn.ReLU() )
            if self.num_layers > 1:
                for i, _ in enumerate(self.decoder_layer_dims):
                    if i == self.num_layers-1:
                        self.decoder_layers.append( nn.Linear( self.decoder_layer_dims[i], self.input_dim )  )
                    else:
                        self.decoder_layers.append( nn.Linear( self.decoder_layer_dims[i], self.decoder_layer_dims[i+1] )  )
                        self.decoder_layers.append( nn.ReLU() )
        else:
            import brevitas.nn as qnn
            self.quant_input = qnn.QuantIdentity(bit_width=self.qpr, return_quant_tensor=True)
            # init encoder layers
            self.encoder_layers = []
            self.encoder_layers.append( qnn.QuantLinear(self.input_dim, self.encoder_layer_dims[0], bias=True, weight_bit_width=self.qpr) )
            self.encoder_layers.append( qnn.QuantReLU(bit_width=self.qpr, return_quant_tensor=True) )
            if self.num_layers > 1:
                for i, _ in enumerate(self.encoder_layer_dims):
                    if i == self.num_layers-1:
                        self.encoder_layers.append( qnn.QuantLinear(self.encoder_layer_dims[i], self.latent_dim, bias=True, weight_bit_width=self.qpr)  )
                    else:
                        self.encoder_layers.append( qnn.QuantLinear(self.encoder_layer_dims[i], self.encoder_layer_dims[i+1], bias=True, weight_bit_width=self.qpr)  )
                        self.encoder_layers.append( qnn.QuantReLU(bit_width=self.qpr, return_quant_tensor=True) )
    
            # init decoder layers
            self.decoder_layers = []
            self.decoder_layers.append( qnn.QuantLinear(self.latent_dim, self.decoder_layer_dims[0], bias=True, weight_bit_width=self.qpr) )
            self.decoder_layers.append( qnn.QuantReLU(bit_width=self.qpr, return_quant_tensor=True) )
            if self.num_layers > 1:
                for i, _ in enumerate(self.decoder_layer_dims):
                    if i == self.num_layers-1:
                        self.decoder_layers.append( qnn.QuantLinear(self.decoder_layer_dims[i], self.input_dim, bias=True, weight_bit_width=self.qpr) )
                    else:
                        self.decoder_layers.append( qnn.QuantLinear(self.decoder_layer_dims[i], self.decoder_layer_dims[i+1], bias=True, weight_bit_width=self.qpr)  )
                        self.decoder_layers.append( qnn.QuantReLU(bit_width=self.qpr, return_quant_tensor=True) )

        self.encoder_layers = nn.ModuleList( self.encoder_layers )
        self.decoder_layers = nn.ModuleList( self.decoder_layers )

        print(self.encoder_layers)
        print(self.decoder_layers)
        
        # send network to device
        self.to( self.device )

        # training
        self.loss = nn.MSELoss()
        self.loss_eval = nn.MSELoss( reduction='none' )
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.loss_hist = []

    def forward(self, x):
        if self.quant:
            x = self.quant_input( x )
        for l in self.encoder_layers:
            x = l(x)
        for l in self.decoder_layers:
            x = l(x)
        return x

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
            #print( "epoch " + str(epoch) + " - " + "loss: " + str( epoch_loss ) )

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
        inv_fpr = 1 / fpr
        target_tpr = 0.3
        index = np.argmin(np.abs(tpr - target_tpr))
        inv_fpr_at_0_3 = inv_fpr[index]
        return tpr, inv_fpr, roc_auc, inv_fpr_at_0_3

    def plot_roc_curve(self, test_dataloader, labels=None, savefig=None):
        eval_losses, eval_labels, roc_auc, inv_fpr_at_0_3 = self.eval_model(test_dataloader)
        fpr, tpr, thresholds = roc_curve(eval_labels, eval_losses)
        fpr = np.nan_to_num(fpr)
        tpr = np.nan_to_num(tpr)
        inv_fpr = 1 / fpr
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        if labels:
            ax.plot(tpr, inv_fpr, color='b', lw=2, label=f'{labels[0]}, AUC={roc_auc:.3f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        else:
            ax.plot(tpr, inv_fpr, color='b', lw=2, label=f'AUC={roc_auc:.2f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        ax.plot([1-i/10000 for i in range(10000)], [1/(1-i/10000) for i in range(10000)], color='gray', linestyle='--')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_s$', fontsize=16)
        ax.set_ylabel(r'$\epsilon_b^{-1}$', fontsize=16)
        ax.set_ylim(1,1000)
        ax.legend(loc="upper right")
        ax.grid(True)
        if savefig:
            fig.savefig(savefig, dpi=500)
        plt.show()

    def plot_roc_curves(self, test_dataloaders, labels, colors=['b','r','g','m'], savefig=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        aucs = []
        for i, test_dataloader in enumerate(test_dataloaders):
            eval_losses, eval_labels, roc_auc, inv_fpr_at_0_3 = self.eval_model(test_dataloader)
            aucs.append(roc_auc)
            fpr, tpr, thresholds = roc_curve(eval_labels, eval_losses)
            fpr = np.nan_to_num(fpr)
            tpr = np.nan_to_num(tpr)
            inv_fpr = 1 / fpr
            if labels:
                ax.plot(tpr, inv_fpr, color=colors[i], lw=2, label=f'{labels[i]}, AUC={roc_auc:.3f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
            else:
                ax.plot(tpr, inv_fpr, color=colors[i], lw=2, label=f'AUC={roc_auc:.2f}, $\epsilon_b^{{(0.3)}}$={inv_fpr_at_0_3:.0f}')
        ax.plot([1-i/10000 for i in range(10000)], [1/(1-i/10000) for i in range(10000)], color='gray', linestyle='--')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_s$', fontsize=16)
        ax.set_ylabel(r'$\epsilon_b^{-1}$', fontsize=16)
        ax.set_ylim(1,1000)
        ax.legend(loc="upper right")
        ax.grid(True)
        for i, label in enumerate(labels):
            print( label+": "+str(aucs[i]) )
        if savefig:
            fig.savefig(savefig, dpi=500)
        plt.show()

    def plot_loss_epochs(self, log_yscale=False, savefig=None, show_plots=True):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_box_aspect(1)
        epochs = [ i for i in range( len(self.loss_hist) ) ]
        ax.plot( epochs, self.loss_hist, color='black', lw=2, label='DNN-AE')
        ax.set_xlabel(r'Epoch', fontsize=16)
        ax.set_ylabel(r'Loss', fontsize=16)
        ax.tick_params(labelsize=14)
        if log_yscale:
            ax.set_yscale('log')
        ax.legend(loc="best", fontsize=16)
        ax.grid(True)
        if savefig:
            fig.savefig(savefig, dpi=500, bbox_inches="tight")
        if show_plots:
            plt.show()

