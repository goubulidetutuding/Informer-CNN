import torch
import torch.nn as nn
from collections.abc import Mapping, Iterable
from genericpath import isfile
import pathlib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import Informer2020_main
from Informer2020_main.utils.timefeatures import time_features

class makeDataset(torch.utils.data.Dataset):
    def __init__(self, x, y,x_mark,y_mark,hycom_sst,hycom_sal,hycom_u,hycom_v):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        self.x_mark = torch.tensor(x_mark)
        self.y_mark = torch.tensor(y_mark)
        self.hycom_sst = torch.tensor(hycom_sst)
        self.hycom_sal = torch.tensor(hycom_sal)
        self.hycom_u = torch.tensor(hycom_u)
        self.hycom_v = torch.tensor(hycom_v)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x_mark = self.x_mark[idx]
        y_mark = self.y_mark[idx]
        hycom_sst = self.hycom_sst[idx]
        hycom_sal = self.hycom_sal[idx]
        hycom_u = self.hycom_u[idx]
        hycom_v = self.hycom_v[idx]
        return x, y, x_mark, y_mark, hycom_sst,hycom_sal,hycom_u,hycom_v
    
class makeDataset_hycom(torch.utils.data.Dataset):
    def __init__(self, x, hycom_x,y):
        self.x = torch.tensor(x)#.float()
        self.hycom_x = torch.tensor(hycom_x)#.float()
        self.y = torch.tensor(y)#.float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        hycom_x = self.hycom_x[idx]
        y = self.y[idx]
        return x,  hycom_x, y
    
def minmax(array):
   ymax = 1
   ymin = 0
   xmax = max(map(max,array))
   xmin = min(map(min,array))

   for i in range(len(array)):
      for j in range(len(array[0])):
         array[i][j] = round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin,3)
   return array



def adam(model, **kwargs):
    return torch.optim.Adam(model.parameters(), **kwargs)



# 对data数组进行还原 数据集中的lat和lon为差值，还原成实际的值
def data_reduction(data,last_lat,last_lon):
    result = data.copy()
    for i in range(data.shape[1]):
        if i % 2 == 0:
            result[:, i] += last_lat
        else:
            result[:, i] += last_lon

    return result

# 对data每一个结果后添加"特征值-2列"(2列为lat和lon)方便后续计算
def data_add_cols(data, n_features):
    data_rows, data_cols = data.shape
    new_data_cols = n_features - 2
    cols_to_add = (data_cols // 2) * new_data_cols
    # 初始化新的数组
    result = np.zeros((data_rows, data_cols + cols_to_add))
    # 逐列复制元素并添加零列
    j = 0
    for i in range(data_cols):
        result[:, j] = data[:, i]
        j += 1
        if (i + 1) % 2 == 0:
            result[:, j:j + new_data_cols] = 0
            j += new_data_cols

    return result

# 制作mark
def data_mark(data_time, data_X, data_y,in_len,label_len):
    data_time = pd.DataFrame(data_time).apply(pd.to_datetime)  # 将日期格式进行转换
    b = []
    for i in range(data_time.shape[1]):  # 对每一列进行处理
        a = data_time.loc[:, i].values
        a = pd.DataFrame(a, columns=['date'])
        a = time_features(a, timeenc=1, freq='h')  # 使用informer的time_features函数对时间进行处理，将每一个时间变成一个一行四列的数组
        b.append(a)
    data_time = np.stack(b, axis=1)  

    # mark是包含一个时间信息(有4列)，和一个经度信息，和一个纬度信息
    data_X_mark = np.concatenate((data_time[:, :in_len, :], data_X[:, :, :2]),
                                 axis=2)  
    print(data_time.shape,data_y.shape)
    data_y_mark = np.concatenate((data_time[:, in_len - label_len:, :], data_y[:, :, :2]),
                                 axis=2)  

    return data_X_mark, data_y_mark

def makeDataset_trace(trainfile_path,time_id_path,n_features,in_len,out_len,nn1,nn2,label_len):
    data = pd.read_csv(trainfile_path, header=None).values
    data_time = pd.read_csv(time_id_path, header=None).values

    n1 = int(len(data) * nn1)
    n2 = int(n1 * nn2)

    data_x = data[:, 0:n_features * in_len]
    data_y = data[:, n_features * in_len:n_features * in_len + out_len]

    last_lat = data_x[:, n_features * (in_len - 1)]
    last_lon = data_x[:, n_features * (in_len - 1) + 1]

    # 对data_y数组进行还原 数据集中的lat和lon为差值，还原成实际的值
    data_y = data_reduction(data=data_y,last_lat=last_lat,last_lon=last_lon)

    # 对data每一个结果后添加"特征值-2列"(2列为lat和lon)方便后续计算
    data_y = data_add_cols(data_y, n_features)

    # 对数据进行归一化
    x_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_x)
    data_x = x_scaler1.transform(data_x)
    y_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_y)
    data_y = y_scaler1.transform(data_y)

    x_data = data_x.reshape((data_x.shape[0], in_len, n_features))  
    y_data = data_y.reshape((data_y.shape[0], int(out_len / 2), n_features))  
    time_data = data_time.reshape((data_time.shape[0], in_len + int(out_len / 2), 2))
    time_data = time_data[:, :, 1]

    data_train_x = x_data[:n1, :, :]
    data_train_y = y_data[:n1, :, :]
    data_train_time = time_data[:n1, :]
    data_test_x = x_data[n1:, :, :]
    data_test_y = y_data[n1:, :, :]
    data_test_time = time_data[n1:, :]

    train_X = data_train_x[:n2, :, :]  
    train_y = data_train_y[:n2, :, :]  
    train_y = np.concatenate((train_X[:, -label_len:, :], train_y), axis=1)  
    print(train_y.shape)
    train_time = data_train_time[:n2, :]  
    train_X_mark, train_y_mark = data_mark(data_time=train_time,
                                           data_X=train_X,data_y=train_y,
                                           in_len=in_len,label_len=label_len)  

    valid_X = data_train_x[n2:, :, :]  
    valid_y = data_train_y[n2:, :, :]  
    valid_y = np.concatenate((valid_X[:, -label_len:, :], valid_y), axis=1)  
    valid_time = data_train_time[n2:, :]  
    valid_X_mark, valid_y_mark = data_mark(data_time=valid_time, data_X=valid_X,data_y=valid_y,
                                           in_len=in_len,label_len=label_len)  

    test_X = data_test_x[:, :, :]  
    test_y = data_test_y[:, :, :]  
    test_y = np.concatenate((test_X[:, -label_len:, :], test_y), axis=1)  
    test_time = data_test_time[:, :]  
    test_X_mark, test_y_mark = data_mark(data_time=test_time, data_X=test_X,data_y=test_y,
                                         in_len=in_len,label_len=label_len)  

    all_data = []
    all_data.append(train_X)
    all_data.append(train_y)
    all_data.append(valid_X)
    all_data.append(valid_y)
    all_data.append(test_X)
    all_data.append(test_y)
    all_data.append(x_scaler1)
    all_data.append(y_scaler1)
    all_data.append(train_X_mark)
    all_data.append(train_y_mark)
    all_data.append(valid_X_mark)
    all_data.append(valid_y_mark)
    all_data.append(test_X_mark)
    all_data.append(test_y_mark)

    return train_X,train_y,valid_X,valid_y,test_X,test_y,x_scaler1,y_scaler1,train_X_mark, train_y_mark,valid_X_mark, valid_y_mark,test_X_mark, test_y_mark,all_data



def makehycom_dataset(path,numbers_pred,nn1,nn2,hycom_x,hycom_y):
    data = pd.read_csv(path,header=None).values
    # data = dd.read_csv(path,header=None)
    
    n1 = int(len(data)*nn1)
    n2 = int(n1*nn2)
    data_train = data[:n1,:]
    data_test = data[n1:,:]

    train_hy = data_train[:n2,]
    valid_hy = data_train[n2:,]
    test_hy  = data_test

    train_hycom_x = train_hy.reshape(train_hy.shape[0],numbers_pred,1,hycom_x,hycom_y)
    valid_hycom_x = valid_hy.reshape(valid_hy.shape[0],numbers_pred,1,hycom_x,hycom_y)
    test_hycom_x = test_hy.reshape(test_hy.shape[0],numbers_pred,1,hycom_x,hycom_y)    

    train_hycom_x=train_hycom_x.transpose(0,1,3,4,2)
    valid_hycom_x=valid_hycom_x.transpose(0,1,3,4,2)
    test_hycom_x=test_hycom_x.transpose(0,1,3,4,2)
    return train_hycom_x,valid_hycom_x,test_hycom_x