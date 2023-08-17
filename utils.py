import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib
from itertools import zip_longest
import seaborn as sns


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self,property):
        with open('dataset\\MOFseq_output.txt', 'r') as f:
            MOF_list = [line.strip() for line in f.readlines()]
        self.MOF_seq = []
        for s in MOF_list:
            word_list = s.split()
            self.MOF_seq.append(word_list)
        dataset = pd.read_csv(f"dataset//mof_output.csv")
        self.MOF_P=list(dataset[property])
        
        with open('dataset\\my_dict_output.json') as f:
            my_dict=json.load(f)
        self.symbol2idx=my_dict['symbol2idx']
        for sublist in self.MOF_seq:
            for i, word in enumerate(sublist):
                    sublist[i] = self.symbol2idx[word]

    def __len__(self):
        return len(self.MOF_seq)

    def __getitem__(self, index):
        MOF_seq = torch.tensor(self.MOF_seq[index],dtype=torch.long)
        P= torch.tensor(self.MOF_P[index],dtype=torch.float)

        return MOF_seq,P

    def get_vocab_size(self):
        return len(self.symbol2idx)


def collate_fn(batch):
    max_seq_len = max(len(seq) for seq,_ in batch)
    padded_seqs = []
    labels = []
    for seq, label in batch:
        seq_len = len(seq)
        padded_seq = torch.zeros(max_seq_len).long()
        padded_seq[:seq_len] = seq
        padded_seqs.append(padded_seq)
        labels.append(label)
    return torch.stack(padded_seqs), torch.tensor(labels)





def plot_results(model, train_loader, test_loader,n):


    model.eval()


    train_pred = []
    train_true = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            train_pred.extend(outputs.tolist())
            train_true.extend(targets.tolist())

    test_pred = []
    test_true = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            test_pred.extend(outputs.tolist())
            test_true.extend(targets.tolist())

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    train_pred_array = np.array(train_pred).flatten()
    test_pred_array = np.array(test_pred).flatten()
    fontsize=20
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(5, 5, hspace=0, wspace=0)
    main_ax = fig.add_subplot(gs[1:5, 0:4])
    right_ax = fig.add_subplot(gs[1:5, 4], sharey=main_ax)
    top_ax = fig.add_subplot(gs[0, 0:4], sharex=main_ax)
    fig_len=max(train_true + test_true)+1000


    main_ax.scatter(train_true, train_pred, label='Training', s=30, c='royalblue', linewidths=0.5)
    main_ax.scatter(test_true, test_pred, label='Test', s=30, c='red', linewidths=0.5)
    main_ax.plot([0, fig_len], [0, fig_len],
                 c='k',linewidth=3)
    main_ax.set_xlim(0,fig_len)
    main_ax.set_ylim(0, fig_len)

    main_ax.set_xlabel(r'ASA (m$^2$/g)', fontsize=35)
    main_ax.set_ylabel(r'Predicted ASA (m$^2$/g)', fontsize=35)

    main_ax.tick_params(axis='both', which='major', labelsize=fontsize,width=3,length=10)

    y_bins = np.linspace(0, fig_len, 20)
    print(max(train_true + test_true))
    right_ax.hist([train_pred_array, test_pred_array], bins=y_bins, orientation='horizontal',
                  color=['royalblue', 'red'], alpha=0.6, edgecolor='black', linewidth=1, density=True,rwidth=5)
    right_ax.set_ylim(0, fig_len)
    right_ax.axis('off')

    x_bins = np.linspace(0, fig_len, 20)
    top_ax.hist([train_true, test_true], bins=x_bins, color=['royalblue', 'red'], alpha=0.6,
                edgecolor='black', linewidth=1, density=True, label=['Train', 'Test'],rwidth=5)
    top_ax.set_xlim(0, fig_len)
    top_ax.axis('off')

    train_r2 = r2_score(train_true, train_pred)
    test_r2 = r2_score(test_true, test_pred)
    train_mae = int(mean_absolute_error(train_true, train_pred))
    test_mae = int(mean_absolute_error(test_true, test_pred))
    train_srcc, _ = spearmanr(train_true, train_pred)
    test_srcc, _ = spearmanr(test_true, test_pred)

    main_ax.text(0.74,0.04,f'Training   \n'f' {train_r2:.2f}\n'f' {train_mae}\n'f' {train_srcc:.2f}', transform=main_ax.transAxes, fontsize=20)
    main_ax.text(0.9, 0.04, f'Test\n'f'{test_r2:.2f}\n'f'{test_mae}\n'f'{test_srcc:.2f}',
                 transform=main_ax.transAxes, fontsize=20)
    main_ax.text(0.6, 0.04, f'     $R\mathrm{{^2}}$:\n'f' MAE:\n'f'SRCC:',
                 transform=main_ax.transAxes, fontsize=20)

    main_ax.text(0.02, 0.9, "MOF-GRU", transform=main_ax.transAxes,
                 fontsize=30, fontweight='bold')
    bwith=2
    main_ax.spines['right'].set_linewidth(bwith)
    main_ax.spines['bottom'].set_linewidth(bwith)
    main_ax.spines['left'].set_linewidth(bwith)
    main_ax.spines['top'].set_linewidth(bwith)
    main_ax.legend(fontsize=20,bbox_to_anchor=(0.32, 0.9))



    plt.tight_layout()
    plt.savefig('tu2//a.tif',dpi=600)
    plt.show()




