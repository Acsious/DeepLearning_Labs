import torch 
import torch.nn as nn 
import pandas as pd
import matplotlib.pyplot as plt

n=26
if(n%2)==1:
    print('Решите задачу классификации покупателей на классы *купит* - *не купит* (3й столбец) по признакам возраст и доход.')
else: 
    print('Решите задачу предсказания дохода по возрасту.')


class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet_regression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size) 
        )
    
    def forward(self, X):
        return self.layers(X)

df = pd.read_csv('dataset_simple.csv')
X_raw = df[['age']].values                  
X = torch.Tensor(X_raw)
X = (X - X.mean()) / X.std()
y = torch.Tensor(df['income'].values).unsqueeze(1)

inputSize   = X.shape[1]    
hiddenSizes = 300            
outputSize  = 1
net = NNet_regression(inputSize, hiddenSizes, outputSize)
lossFn    = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.002) 

epochs = 1000
for i in range(epochs):
    pred = net(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1) % 200 == 0:
        print(f'Ошибка на {i+1} итерации: {loss.item():.2f}')

with torch.no_grad():
    pred = net(X)
    mae = torch.mean(torch.abs(pred - y))
    print(f'\nСредняя абсолютная ошибка (MAE): {mae.item():.2f}')

plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['income'], color='steelblue', label='Реальные данные', alpha=0.7)

sorted_idx = torch.argsort(torch.Tensor(X_raw.squeeze()))
plt.plot(X_raw[sorted_idx], pred[sorted_idx].numpy(), color='red', linewidth=2.5, label='Нейронная сеть')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Предсказание дохода по возрасту')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()




























































































# Пасхалка, кто найдет и сможет объяснить, тому +
# X = np.hstack([np.ones((X.shape[0], 1)), df.iloc[:, [0]].values])

# y = df.iloc[:, -1].values

# w = np.linalg.inv(X.T @ X) @ X.T @ y

# predicted = X @ w

# print(predicted)


























