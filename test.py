import csv
import shap
import torch
from torch.utils.data import Dataset, DataLoader
from utils import MyDataset,collate_fn,plot_results


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A=['CH4ABL','N2ABL','Sa','PLD','LCD','Density','POAVAg','ASA','vASA','gASA']
A=['gASA']
for a in A:
    torch.manual_seed(42)
    # best_model=torch.load(f'my_models\\new\\biGRU_{a}_model222.pth').to(device)
    # best_model = torch.load(f'my_models\\training_set\\biGRU_{a}_1.0_model_xin.pth').to(device)
    best_model = torch.load(f'my_models\\new\\biGRU_{a}_model_ep_40_em_80_hd200.pth').to(device)
    # best_model=torch.load(f'my_models\\new\\biGRU_POAVAg_model.pth').to(device)
    # best_model=torch.load(f'my_models\\new\\biGRU_POAVAg_model.pth').to(device)
    # best_model = torch.load(f'my_models\\training_set\\biGRU_Sa_1.0_model.pth').to(device)
    # input=torch.tensor([1,2,3,4,5,5]).unsqueeze(0).to(device)
    # print(input.shape)
    # lens=torch.tensor(6).unsqueeze(0).to(device)
    # print(lens.shape)
    # output=best_model(input,lens)
    # print(output)
    # 加载训练好的模型和数据
    dataset=MyDataset(a)

    # 划分训练集和测试集
    batch_size=100
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
    test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)

    # 绘制训练集测试集R方
    plot_results(best_model,train_loader,test_loader,a)








# print(f'dataset[0]是{dataset[0]}')
# print(f'dataset[0][0]是{dataset[0][0]}')
# print(f'dataset[0][0]的shape是{dataset[0][0].shape}')
# print(f'dataset[0][0].unsqueeze(0)是{dataset[0][0].unsqueeze(0)}')
#
#
#
# # 计算 SHAP 值
# best_model.train()
# explainer = shap.Explainer(best_model,input_data)
# shap_values = explainer()
# shap.plots.text(shap_values)
# shap.plots.waterfall(shap_values[0])
# print(shap_values)
#
# # 打印每个单词的 SHAP 值
# for i in range(len(dataset[0][0])):
#     print(f"Word '{dataset.vocab.itos[dataset[0][0][i]]}': {shap_values[0][i]}")
# set model to training mode
# best_model.train()
#
# # create explainer
# explainer = shap.DeepExplainer(best_model, input_data,training=True)
#
#
# # calculate SHAP values
# shap_values = explainer.shap_values(input_data)
#
# # print results
# print(shap_values)

# 把预测值和目标值写进csv
# list1=[]
# list2=[]
# for data,target in train_dataset:
#     data=data.to(device)
#     list1.append(float(target))
#     data=data.unsqueeze(0)
#     output=best_model(data)
#     list2.append(float(output))
# with open('data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Column 1', 'Column 2'])
#     for row in zip(list1, list2):
#         writer.writerow(row)
