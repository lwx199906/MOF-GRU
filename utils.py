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
        # self.name=list(dataset['Name'])
        # self.node=list(dataset['node'])
        # self.topo=list(dataset['topo'])
        # self.indexs=list(dataset['index'])
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
        # nodes=self.node[index]
        # topos=self.topo[index]
        # names = self.name[index]
        # ind=self.indexs[index]
        return MOF_seq,P

    def get_vocab_size(self):
        return len(self.symbol2idx)

class MyDataset_test(Dataset):
    def __init__(self,property):
        with open('dataset\\MOFseq_test.txt', 'r') as f:
            MOF_list = [line.strip() for line in f.readlines()]
        self.MOF_seq = []
        for s in MOF_list:
            word_list = s.split()
            self.MOF_seq.append(word_list)
        dataset = pd.read_csv(f"dataset//mof_output_test.csv")
        self.MOF_P=list(dataset[property])
        # self.name=list(dataset['Name'])
        # self.node=list(dataset['node'])
        # self.topo=list(dataset['topo'])
        # self.indexs=list(dataset['index'])
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
        # nodes=self.node[index]
        # topos=self.topo[index]
        # names = self.name[index]
        # ind=self.indexs[index]
        return MOF_seq,P

    def get_vocab_size(self):
        return len(self.symbol2idx)


class MyDataset2(Dataset):
    def __init__(self,property):
        with open('dataset\\MOFseq_dia.txt', 'r') as f:
            MOF_list = [line.strip() for line in f.readlines()]
        self.MOF_seq = []
        for s in MOF_list:
            word_list = s.split()
            self.MOF_seq.append(word_list)
        dataset = pd.read_csv(f"dataset//dia.csv")
        self.MOF_P=list(dataset[property])
        # self.indexs=list(dataset['index'])
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
        # ind=self.indexs[index]
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


class EarlyStoppingCallback:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False

    def __call__(self, model, epoch, train_loss, val_loss):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif val_loss > self.best_val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping on epoch {epoch}")
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0

    def should_stop(self):
        return self.early_stop


def plot_results(model, train_loader, test_loader,n):
    # # 将模型设置为评估模式
    # model.eval()
    #
    # # 计算训练集的预测值和真实值
    # train_pred = []
    # train_true = []
    # with torch.no_grad():
    #     for inputs, targets in train_loader:
    #         inputs=inputs.to(device)
    #         targets=targets.to(device)
    #         outputs = model(inputs)
    #         train_pred.extend(outputs.tolist())
    #         train_true.extend(targets.tolist())
    #
    # # 计算测试集的预测值和真实值
    # test_pred = []
    # test_true = []
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs=inputs.to(device)
    #         targets=targets.to(device)
    #         outputs = model(inputs)
    #         test_pred.extend(outputs.tolist())
    #         test_true.extend(targets.tolist())
    #
    # # # 使用zip_longest对齐列表，并用None填充缺失的元素
    # # aligned_lists = zip_longest(train_true, train_pred, test_true, test_pred, fillvalue=None)
    # #
    # # # 创建DataFrame
    # # df = pd.DataFrame(aligned_lists, columns=['Column1', 'Column2', 'Column3', 'Column4'])
    # #
    # # # 将DataFrame写入CSV文件
    # # df.to_csv('output2345.csv', index=False)
    #
    # # plt.figure(figsize=(6,6))
    # # train_pred = [0 if x < 0 else x for x in train_pred]
    # # test_pred = [0 if x < 0 else x for x in test_pred]
    #
    # # 绘制训练集和测试集的预测值和真实值的图像
    # # plt.scatter(train_true, train_pred, label='Train', s=120, c='royalblue',
    # #             edgecolors='k',
    # #             # alpha=0.8,
    # #             linewidths=0.5
    # #             )
    # # plt.scatter(test_true, test_pred, label='Test', s=120, c='orange',
    # #             edgecolors='k',
    # #             # alpha=0.8,
    # #             linewidths=0.5
    # #             )
    # # matplotlib.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    # train_pred_array = np.array(train_pred).flatten()
    # test_pred_array = np.array(test_pred).flatten()
    #
    # plt.scatter(train_true, train_pred, label='Train', s=10, c='royalblue',
    #             # alpha=0.8,
    #             linewidths=0.5
    #             )
    # plt.scatter(test_true, test_pred, label='Test', s=10, c='red',
    #             # alpha=0.8,
    #             linewidths=0.5
    #             )
    # # 绘制横轴的数据分布线
    #
    # # plt.plot([min(train_true + test_true), max(train_true + test_true)], [min(train_true + test_true), max(train_true + test_true)],
    # #          # linestyle='--',
    # #          c='k',
    # #          label='Ideal')
    # plt.plot([0, max(train_true + test_true)+0.125], [0, max(train_true + test_true)+0.125],
    #          # linestyle='--',
    #          c='k',
    #          label='Ideal')
    # # 绘制边际条形图（横轴）
    # x_bins = np.linspace(0, max(train_true + test_true) + 0.125, 20)  # 设置20个条形的区间
    # plt.hist([train_true, test_true], bins=x_bins, color=['royalblue', 'red'], alpha=0.6,
    #          edgecolor='black', linewidth=1, density=True, label=['Train', 'Test'])
    #
    # # 绘制边际条形图（纵轴）
    # y_bins = np.linspace(0, max(train_true + test_true) + 0.125, 20)  # 设置20个条形的区间
    # plt.hist([train_pred_array, test_pred_array], bins=y_bins, orientation='horizontal',
    #          color=['royalblue', 'red'], alpha=0.6, edgecolor='black', linewidth=1, density=True)
    # plt.xlim(0, max(train_true + test_true)+0.125)  # 设置 x 坐标轴从 0 开始
    # plt.ylim(0, max(train_true + test_true)+0.125)
    # # plt.xlim(bottom=0)
    # # plt.ylim(bottom=0)
    # # plt.xlabel(f'LCD (\u00C5)',fontsize=15)
    # # plt.ylabel(f'Predicted LCD (\u00C5)',fontsize=15)
    # # plt.xlabel(f'Porosity',fontsize=15)
    # # plt.ylabel(f'Predicted Porosity',fontsize=15)
    # # plt.xlabel(f'PV ($cm^3/g$)',fontsize=15)
    # # plt.ylabel(f'Predicted PV ($cm^3/g$)',fontsize=15)
    # # plt.xlabel(f'Density (g/$cm^3$)',fontsize=15)
    # # plt.ylabel(f'Predicted Density (g/$cm^3$)',fontsize=15)
    # # plt.xlabel(f'Simulated N$_{{CH_4}}$(mol/kg)',fontsize=20,fontname = 'Times New Roman')
    # # plt.xlabel(r'Simulated N$_\text{{CH4}}$ (mol/kg)', fontsize=20)
    # #
    # # plt.ylabel(f'Predicted $N_{{CH_4}}$(mol/kg)',fontsize=20)
    # plt.xlabel(r'Simulated CH$_4$/N$_2$ selectivity', fontsize=20)
    #
    # plt.ylabel(r'Predicted CH$_4$/N$_2$ selectivity',fontsize=20)
    # # plt.xlabel(f'ASA ($m^2/g$)',fontsize=15)
    # # plt.ylabel(f'Predicted ASA ($m^2/g$)',fontsize=15)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # # plt.xticks([0, 2000, 4000, 6000, 8000],fontsize=12)
    # # plt.yticks([0, 2000, 4000, 6000, 8000],fontsize=12)
    # # plt.xlabel(f'Simulated $S_{{CH_4/N_2}}$',fontsize=15)
    # # plt.ylabel(f'MOF-GRU Predicted $S_{{CH_4/N_2}}$',fontsize=15)
    # # plt.xlabel(f'Simulated',fontsize=15)
    # # plt.ylabel(f'MOFseq Predicted',fontsize=15)
    # # plt.legend()
    #
    #
    #
    # # 计算训练集和测试集的 R 方
    # train_r2 = r2_score(train_true, train_pred)
    # test_r2 = r2_score(test_true, test_pred)
    # train_mae = mean_absolute_error(train_true, train_pred)
    # test_mae = mean_absolute_error(test_true, test_pred)
    # train_srcc, _ = spearmanr(train_true, train_pred)
    # test_srcc, _ = spearmanr(test_true, test_pred)
    # # print(f'Train R2 score: {train_r2:.4f}')
    # # print(f'Test R2 score: {test_r2:.4f}')
    # plt.text(max(train_true + test_true), min(train_true + test_true),
    #          f'       Train     Test\n'
    #          f'  R$^2$:  {train_r2:.2f}     {test_r2:.2f}\n'
    #          f' MAE:  {train_mae:.2f}     {test_mae:.2f}\n'
    #          # f'MAE:  {int(train_mae)}      {int(test_mae)}\n'
    #          f'SRCC:  {train_srcc:.2f}     {test_srcc:.2f}', ha='right',
    #          va='bottom',
    #          fontsize=20,
    #          )
    #
    #
    #
    #
    # plt.text(0.05, 0.95, "MOF-GRU", transform=plt.gca().transAxes,
    #          fontsize=30, fontweight='bold', va='top', ha='left',fontname = 'Times New Roman')
    # print(max(train_true + test_true))
    # print(min(train_true + test_true))
    # # plt.text(max(train_true + test_true),min(train_true + test_true),
    # #          f'       Train     Test\n'
    # #          f' R2:   {train_r2:.2f}     {test_r2:.2f}\n'
    # #          f'  MAE:  {train_mae:.2f}  {test_mae:.2f}\n'
    # #          f'SRCC:  {train_srcc:.2f}     {test_srcc:.2f}', ha='right',
    # #          va='bottom',
    # #          fontsize=12
    # #          )
    # # plt.title(f'biGRU predicts {n}')
    # # plt.ylim(bottom=0)
    # plt.tight_layout()  # 自动调整布局，确保横轴完整显示
    # # plt.savefig(f'tu2\\预测对比\\MOFseq_predicted_{n}_new.tif',dpi=600)
    # plt.show()


    model.eval()

    # 计算训练集的预测值和真实值
    train_pred = []
    train_true = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            train_pred.extend(outputs.tolist())
            train_true.extend(targets.tolist())

    # 计算测试集的预测值和真实值
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
    # plt.rcParams["text.usetex"] = True
    train_pred_array = np.array(train_pred).flatten()
    test_pred_array = np.array(test_pred).flatten()
    fontsize=20
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(5, 5, hspace=0, wspace=0)
    # gs = fig.add_gridspec(4, 4)
    main_ax = fig.add_subplot(gs[1:5, 0:4])
    right_ax = fig.add_subplot(gs[1:5, 4], sharey=main_ax)
    top_ax = fig.add_subplot(gs[0, 0:4], sharex=main_ax)
    fig_len=max(train_true + test_true)+1000


    # 主图
    main_ax.scatter(train_true, train_pred, label='Training', s=30, c='royalblue', linewidths=0.5)
    main_ax.scatter(test_true, test_pred, label='Test', s=30, c='red', linewidths=0.5)
    main_ax.plot([0, fig_len], [0, fig_len],
                 c='k',linewidth=3)
    main_ax.set_xlim(0,fig_len)
    main_ax.set_ylim(0, fig_len)
    # main_ax.set_xlabel(r'Simulated CH$_4$/N$_2$ Selectivity', fontsize=fontsize)
    # main_ax.set_ylabel(r'Predicted CH$_4$/N$_2$ Selectivity', fontsize=fontsize)
    # main_ax.set_xlabel(r'Simulated $S_{{\mathrm{{CH_4/N_2}}}}$', fontsize=35)
    # main_ax.set_ylabel(r'Predicted $S_{{\mathrm{{CH_4/N_2}}}}$', fontsize=35)
    # main_ax.set_xlabel(r'Simulated $N_\mathrm{{N_2}}$(mol/kg)', fontsize=35)
    # main_ax.set_ylabel(r'Predicted $N_\mathrm{{N_2}}$(mol/kg)', fontsize=35)
    # main_ax.set_xlabel('PLD (\u00C5)', fontsize=35)
    # main_ax.set_ylabel('Predicted PLD (\u00C5)', fontsize=35)
    # main_ax.set_xlabel(r'Density (g/cm$^3$)', fontsize=35)
    # main_ax.set_ylabel(r'Predicted Density (g/cm$^3$)', fontsize=35)
    # main_ax.set_xlabel(r'PV (cm$^3$/g)', fontsize=35)
    # main_ax.set_ylabel(r'Predicted PV (cm$^3$/g)', fontsize=35)
    main_ax.set_xlabel(r'ASA (m$^2$/g)', fontsize=35)
    main_ax.set_ylabel(r'Predicted ASA (m$^2$/g)', fontsize=35)
    # main_ax.set_xlabel(r'Density (g/cm$^3$)', fontsize=35)
    # main_ax.set_ylabel(r'Predicted Density (g/cm$^3$)', fontsize=35)


    # main_ax.set_ylabel(r'Predicted S$_{{CH_4/N_2}}$', fontsize=fontsize)
    main_ax.tick_params(axis='both', which='major', labelsize=fontsize,width=3,length=10)
    # plt.xticks([0,2,4,6,8,10,12,14],)
    # plt.yticks([0, 2, 4, 6, 8, 10, 12, 14])

    # 右侧边际条形图（纵向）
    y_bins = np.linspace(0, fig_len, 20)
    print(max(train_true + test_true))
    right_ax.hist([train_pred_array, test_pred_array], bins=y_bins, orientation='horizontal',
                  color=['royalblue', 'red'], alpha=0.6, edgecolor='black', linewidth=1, density=True,rwidth=5)
    right_ax.set_ylim(0, fig_len)
    right_ax.axis('off')

    # 上方边际条形图（横向）
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
    # main_ax.text(0.5, 0.01,
    #              f'       Train     Test\n'
    #              f'  R$^2$:  {train_r2:.2f}     {test_r2:.2f}\n'
    #              f' MAE:  {train_mae:.2f}     {test_mae:.2f}\n'
    #              f'SRCC:  {train_srcc:.2f}     {test_srcc:.2f}', transform=main_ax.transAxes, fontsize=fontsize)
    # main_ax.text(0.74,0.04,f'Training   \n'f' {train_r2:.2f}\n'f' {train_mae:.2f}\n'f' {train_srcc:.2f}', transform=main_ax.transAxes, fontsize=20)
    # main_ax.text(0.9, 0.04, f'Test\n'f'{test_r2:.2f}\n'f'{test_mae:.2f}\n'f'{test_srcc:.2f}',
    #              transform=main_ax.transAxes, fontsize=20)
    # main_ax.text(0.6, 0.04, f'     $R\mathrm{{^2}}$:\n'f' MAE:\n'f'SRCC:',
    #              transform=main_ax.transAxes, fontsize=20)
    main_ax.text(0.74,0.04,f'Training   \n'f' {train_r2:.2f}\n'f' {train_mae}\n'f' {train_srcc:.2f}', transform=main_ax.transAxes, fontsize=20)
    main_ax.text(0.9, 0.04, f'Test\n'f'{test_r2:.2f}\n'f'{test_mae}\n'f'{test_srcc:.2f}',
                 transform=main_ax.transAxes, fontsize=20)
    main_ax.text(0.6, 0.04, f'     $R\mathrm{{^2}}$:\n'f' MAE:\n'f'SRCC:',
                 transform=main_ax.transAxes, fontsize=20)
    # main_ax.text(0.05, 0.80, "MOF-GRU", transform=plt.gca().transAxes,
    #              fontsize=30, fontweight='bold')
    main_ax.text(0.02, 0.9, "MOF-GRU", transform=main_ax.transAxes,
                 fontsize=30, fontweight='bold')
    bwith=2
    main_ax.spines['right'].set_linewidth(bwith)
    main_ax.spines['bottom'].set_linewidth(bwith)
    main_ax.spines['left'].set_linewidth(bwith)
    main_ax.spines['top'].set_linewidth(bwith)
    main_ax.legend(fontsize=20,bbox_to_anchor=(0.32, 0.9))



    plt.tight_layout()
    plt.savefig('tu2//asa预测最新.tif',dpi=600)
    plt.show()




