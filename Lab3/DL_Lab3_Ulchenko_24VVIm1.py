import torch
import torch.nn as nn
import pandas as pd
import numpy as np

################ Задание 1 ###################
x = torch.randint(5, [5])
print(x) 

x = x.to(dtype=torch.float32)
print(x)

x **= 3
print(x)

x*=torch.randint(10,[1])
print(x)

x = torch.exp(x)
print(x)

################ Задание 2 ###################
    
df = pd.read_csv('data.csv')

X = torch.Tensor(df.iloc[:, [0, 1, 3]].values)   # (150, 3)
y = torch.Tensor(np.where(df.iloc[:, 4].values == "Iris-setosa", 1.0, 0.0)).unsqueeze(1)  # (150, 1)

# создадим 3 сумматора без функци активации, это называется полносвязный слой (fully connected layer)
# Отсутствие фунций активаци на выходе сумматора эквивалетно наличию  линейной активации
linear = nn.Linear(3, 1)

# при создании веса и смещения инициализируются автоматически
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# выберем вид функции ошибки и оптимизатор
# фунция ошибки показывает как сильно ошибается наш алгоритм в своих прогнозах
#lossFn = nn.MSELoss() # MSE - среднеквадратичная ошибка, вычисляется как sqrt(sum(y^2 - yp^2))
lossFn = nn.BCEWithLogitsLoss() 

# создадим оптимизатор - алгоритм, который корректирует веса наших сумматоров (нейронов)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01) # lr - скорость обучения

# прямой проход (пресказание) выглядит так:
yp = linear(X)

# имея предсказание можно вычислить ошибку
loss = lossFn(yp, y)
print('Ошибка: ', loss.item())

# и сделать обратный проход, который вычислит градиенты (по ним скорректируем веса)
loss.backward()

# градиенты по параметрам
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# далее можем сделать шаг оптимизации, который изменит веса 
# на сколько изменится каждый вес зависит от градиентов и скорости обучения lr
optimizer.step()

# итерационно повторяем шаги
# в цикле (фактически это и есть алгоритм обучения):
for i in range(0,150):
    pred = linear(X)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    pred = linear(X)
    prob = torch.sigmoid(pred)                    
    predicted_class = (prob > 0.5).float()       
    
    correct = (predicted_class == y).sum().item()
    accuracy = correct / len(y)
    errors = len(y) - correct

    print('\nОбучение завершено!')
    print(f'Точность: {accuracy*100:.2f}%')
    print(f'Ошибок: {errors} из {len(y)}')
    print(f'Веса нейрона: {linear.weight.data.squeeze().numpy()}')
    print(f'Смещение (bias): {linear.bias.data.item():.4f}')