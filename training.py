import time
import torch
import torch.nn as nn
from models import GRUModel,LSTMModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import MyDataset,collate_fn,EarlyStoppingCallback


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

A=['CH4ABL','N2ABL','PLD','LCD','Density','Porosity','PV','gASA']
for a in A:
    torch.manual_seed(42)
    print('准备数据...')
    dataset=MyDataset(a)


    print('划分训练集测试集...')
    batch_size=100
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
    test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)
    epo=[20,40,60]
    em = [20, 40, 60, 80]
    hd = [100, 150, 200]
    for ep in epo:
        for e in em:
            for h in hd:
                print('模型初始化...')
                vocab_size=dataset.get_vocab_size()+1
                embedding_size=e
                hidden_size=h
                num_layers=1
                model=GRUModel(vocab_size, embedding_size, hidden_size, num_layers).to(device)
                # 训练模型
                epoch=ep
                best_loss=1000000000000000000
                # criterion = nn.L1Loss()
                criterion = nn.MSELoss()
                early_stop_callback = EarlyStoppingCallback(patience=10)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                print('开始训练循环...')
                Start_time=time.time()
                for i in range(epoch):
                    start_time=time.time()
                    print(f'第{i+1}轮训练开始：')
                    model.train()
                    train_step=0
                    total_train_loss=0
                    for (batch,P) in train_loader:
                        batch=batch.to(device)
                        P = P.to(device)
                        x_hat=model(batch)
                        # x_hat=x_hat.squeeze()
                        loss=criterion(x_hat.squeeze(),P).to(device)
                        # loss = criterion(torch.log(x_hat + 1), torch.log(P + 1)).to(device)
                        total_train_loss +=loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_step +=1
                    print(f'训练集上的loss：{total_train_loss/train_step}')
                    end_time=time.time()
                    print(f'第{i+1}轮训练结束,用时{end_time-start_time}秒')
                torch.save(model,f'my_models\\new\\biGRU_{a}_model_ep_{ep}_em_{e}_hd{h}.pth')
                End_time = time.time()
                print(f'{epoch}轮训练结束，总用时{(End_time - Start_time) / 60}分钟')
