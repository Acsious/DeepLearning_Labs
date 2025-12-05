import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint

################ Задание 1 ###################
random_list = [randint(0, 10) for _ in range(5)]
print("Сгенерированный список:")
print(random_list)

even_sum = 0
for num in random_list:
    if num % 2 == 0:  
        even_sum += num

print("Сумма чётных чисел в списке:", even_sum)


################ Задание 2 ###################
df = pd.read_csv('data.csv')
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [1, 2, 3]].values  

plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x', label='Другие')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.title('Проекция на первые два признака из трёх')
plt.show()

def neuron(w, x):
    if w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2] >= 0:
        return 1
    else:
        return -1

w_test = np.array([0, 0.1, 0.4, 0.2])
print("Тест нейрона на втором примере:", neuron(w_test, X[1]))

np.random.seed(42)
w = np.random.random(4)
eta = 0.01  # скорость обучения
w_iter = [] # пустой список, в него будем добавлять веса, чтобы потом построить график
for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w,xi)   
    w[1:] += (eta * (target - predict)) * xi # target - predict - это и есть ошибка
    w[0] += eta * (target - predict)
    # каждую 10ю итерацию будем сохранять набор весов в специальном списке
    if(j%10==0):
        w_iter.append(w.tolist())


# посчитаем ошибки
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w,xi) 
    sum_err += (target - predict)/2
    print("ошибка", sum_err)

print("Всего ошибок: ", sum_err)
print("Финальные веса:", w)


# попробуем визуализировать процесс обучения
xl=np.linspace(min(X[:,0]), max(X[:,0])) # диапазон координаты x для построения линии

# построим сначала данные на плоскости
plt.figure
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x') 

# потом в цикле будем брать набор весов из сохраненного списка и по нему строить линию
for i,w in zip(range(len(w_iter)), w_iter):
    yl = -(xl*w[1]+w[0])/w[2] # уравнение линии
    plt.plot(xl, yl) # строим разделяющую границу
    plt.text(xl[-1], yl[-1], i, dict(size=10, color='gray')) # подписываем номер линии
    plt.pause(1)
    
plt.text(xl[-1]-0.3, yl[-1], 'END', dict(size=14, color='red'))
plt.show() 