import torch
import torch.nn as nn
from models import GRUModel
from torch.utils.data import Dataset, DataLoader
from utils import MyDataset,collate_fn,EarlyStoppingCallback

torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型和数据
print('准备数据...')
dataset=MyDataset('N2ABL')
# 'Density'

# 划分训练集和测试集
print('划分训练集测试集...')
batch_size=100
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)


# 模型初始化
vocab_size=dataset.get_vocab_size()+1
Hidden_dim=[100,200,300]
Embedding_dim=[20,40,60,80,100]

for hidden_size in Hidden_dim:
    for embedding_size in Embedding_dim:
        model=GRUModel(vocab_size, embedding_size, hidden_size, num_layers=1).to(device)
        # 训练模型
        epoch=50
        best_loss=1000000000000000000
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        early_stop_callback = EarlyStoppingCallback(patience=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for i in range(epoch):
            print(f'第{i+1}轮训练开始：')
            model.train()
            train_step=0
            total_train_loss=0
            for (batch,P) in train_loader:
                batch=batch.to(device)
                P = P.to(device)
                x_hat=model(batch)
                loss=criterion(x_hat.squeeze(),P).to(device)
                # loss = criterion(torch.log(x_hat + 1), torch.log(P + 1)).to(device)
                total_train_loss +=loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_step +=1
            print(f'训练集上的loss：{total_train_loss/train_step}')
            model.eval()
            test_step=0
            total_test_loss=0
            with torch.no_grad(): # 保证不会对模型进行调优
                for (batch,P) in test_loader:
                    batch = batch.to(device)
                    P = P.to(device)
                    x_hat= model(batch)
                    loss = criterion(x_hat.squeeze(), P).to(device)
                    # loss = criterion(torch.log(x_hat + 1), torch.log(P + 1)).to(device)
                    total_test_loss += loss
                    test_step+=1
                print(f'测试集上的loss：{total_test_loss / test_step}')
            if total_test_loss/ test_step < best_loss:
                best_loss = total_test_loss/ test_step
                best_model=model
        with open('my_models\\tuning_record\\N2ABL_tuning.txt', "a") as f:
            f.write(f"gru模型预测氮气吸附，隐层维度：{hidden_size}\t,embedding维度：{embedding_size}\t,loss:{best_loss}\n")