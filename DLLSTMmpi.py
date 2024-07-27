import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from math import sqrt
from mpi4py import MPI

# 初始化MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 数据加载
if rank == 0:
    from google.colab import drive
    drive.mount('/content/drive')
    data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/ANLY530/GOOG725.csv")
    data = data['Close'].values.reshape(-1, 1)
else:
    data = None

# 广播数据到所有进程
data = comm.bcast(data, root=0)

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 创建训练和测试数据集
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# 划分数据集到不同的进程
num_train_samples = len(X_train)
num_test_samples = len(X_test)
train_samples_per_proc = num_train_samples // size
test_samples_per_proc = num_test_samples // size

X_train_local = X_train[rank * train_samples_per_proc:(rank + 1) * train_samples_per_proc]
Y_train_local = Y_train[rank * train_samples_per_proc:(rank + 1) * train_samples_per_proc]
X_test_local = X_test[rank * test_samples_per_proc:(rank + 1) * test_samples_per_proc]
Y_test_local = Y_test[rank * test_samples_per_proc:(rank + 1) * test_samples_per_proc]

# 调整数据形状
X_train_local = X_train_local.reshape(X_train_local.shape[0], X_train_local.shape[1], 1)
X_test_local = X_test_local.reshape(X_test_local.shape[0], X_test_local.shape[1], 1)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 记录训练时间
start_time = time.time()
# 模型训练
model.fit(X_train_local, Y_train_local, batch_size=5, epochs=10)
training_time = time.time() - start_time
print(f'Rank {rank} Training Time: {training_time:.2f} seconds')

# 记录预测时间
start_time = time.time()
# 预测
train_predict_local = model.predict(X_train_local)
test_predict_local = model.predict(X_test_local)
testing_time = time.time() - start_time
print(f'Rank {rank} Testing Time: {testing_time:.2f} seconds')

# 反归一化
train_predict_local = scaler.inverse_transform(train_predict_local)
test_predict_local = scaler.inverse_transform(test_predict_local)

# 收集所有进程的预测结果
train_predict = np.zeros((num_train_samples, 1))
test_predict = np.zeros((num_test_samples, 1))

comm.Gather(train_predict_local, train_predict, root=0)
comm.Gather(test_predict_local, test_predict, root=0)

# 计算全局的MSE和RMSE
if rank == 0:
    # 将预测结果存储到CSV文件
    train_predict_df = pd.DataFrame(train_predict, columns=['Train Predict'])
    test_predict_df = pd.DataFrame(test_predict, columns=['Test Predict'])

    train_predict_df.to_csv('/content/drive/My Drive/Colab Notebooks/train_predict.csv', index=False)
    test_predict_df.to_csv('/content/drive/My Drive/Colab Notebooks/test_predict.csv', index=False)

    # 计算MSE和RMSE
    train_score = mean_squared_error(data[time_step:len(train_predict) + time_step], train_predict)
    train_rmse = sqrt(train_score)
    print(f'Train Score: MSE = {train_score}, RMSE = {train_rmse}')

    test_score = mean_squared_error(data[len(train_predict) + (time_step * 2) + 1:len(data) - 1], test_predict)
    test_rmse = sqrt(test_score)
    print(f'Test Score: MSE = {test_score}, RMSE = {test_rmse}')
