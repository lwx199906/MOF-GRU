import csv
import shap
import torch
from torch.utils.data import Dataset, DataLoader
from utils import MyDataset,collate_fn,plot_results


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

A=['CH4ABL','N2ABL','Sa','PLD','LCD','Density','POAVAg','ASA','vASA','gASA']

for a in A:
    torch.manual_seed(42)

    best_model = torch.load(f'my_models\\new\\biGRU_{a}_model_ep_40_em_80_hd200.pth').to(device)

    dataset=MyDataset(a)


    batch_size=100
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
    test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)


    plot_results(best_model,train_loader,test_loader,a)
