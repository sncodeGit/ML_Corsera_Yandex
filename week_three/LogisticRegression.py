GRAD_STEP = 0.1 #Градиентный шаг
ACCEPT_DIST = 1e-5 #Приемлемое расстояние между векторами весов на итерации (до и после)
MAX_ITER = 10000 #Максисальное количество итераций на градиентном спуске
L2_REG_COEF = 10

import numpy as np
import pandas as pd

#Метод градиентного спуска для логистической регрессии без L2-регуляризации
#Подбор значения вектора весов (w1, w2)
#С возможностью использования L2-регуляризации
def grad_descent(C = 0):
    w1 = w2 = 0 #Начальное значение вектора весов
    w1_old = w2_old = 1 #Чтобы по разности новых и старых весов while() 
    #пропустил на 1-ую итерацию
    object_count = len(df)
    i = 0 #Счетчик итераций
    while (i < MAX_ITER and 
           np.sqrt(abs(w1_old - w1)**2 + abs(w2_old - w2)**2) > ACCEPT_DIST):
        i += 1
        w1_old = w1
        w2_old = w2
        sum_1 = sum_2 = 0
        #Сумма по количеству элементов из формулы градиентного шага
        for j in range(0, object_count):
            sum_all = df[0][j]*(1 - 1/(1 + np.exp(-1*df[0][j]*(w1*df[1][j] + w2*df[2][j]))))
            sum_1 = sum_1 + sum_all*df[1][j]
            sum_2 = sum_2 + sum_all*df[2][j]
        #Делаем градиентный шаг
        w1 = w1 + GRAD_STEP/object_count*sum_1 - GRAD_STEP*C*w1
        w2 = w2 + GRAD_STEP/object_count*sum_2 - GRAD_STEP*C*w2
        
    if (i == 10000):
        print('Веса не сошлись :(')
        print('Расстояние: ', 
              np.sqrt(abs(w1_old - w1)**2 + abs(w2_old - w2)**2))
    else:
        print('Сошлось :)')
        print('Количество итераций:', i)
        
    weights = [0] * 2 #Создаём массив из двух нулевых элементов
    weights[0] = w1
    weights[1] = w2
    return weights
        
df = pd.read_csv('../data/data_logistic.csv', header=None)

weights_none = grad_descent() #Вектор весов без регуляризации
weights_l2 = grad_descent(L2_REG_COEF) #Вектор весов с L2-регуляризацией

#Используем метрику оценки качества AUC-ROC
#AUC-ROC - площадь под кривой ошибок (кривая от (0,0) до (1,1)) 
#[больше - лучше, а кривая в идеале стремится к (0,1), т.к. мы хотим максимизировать TPR,
#а FPR минимизировать]
#в координатах TPR(R - Rate) = TP/(TP+FN) и FPR = FP/(FP+TN)
#Если ошибок нет (FPR = 0, TPR = 1) - площадь равна 1
#Если вероятность выдаётся случайно, то площадь будет стремиться к 0.5 
#т.к. примерно одинаковое количество FP и TP

from sklearn.metrics import roc_auc_score

object_count = len(df)
#Посчитаем вероятности принадлежности к первому классу для каждого классификатора
probability_none = [0] * object_count
probability_l2 = [0] * object_count
for j in range(0, object_count):
    probability_none[j] = 1/(1 + 
                    np.exp(-1*(weights_none[0]*df[1][j] + weights_none[1]*df[2][j])))
    probability_l2[j] = 1/(1 + np.exp(-1*(weights_l2[0]*df[1][j] + weights_l2[1]*df[2][j])))
#В формуле в знаменателе заменим значение класса на 1, чтобы посчитать вероятность
#см. формулу на Courser'е

print(roc_auc_score(df[0], probability_none))
print(roc_auc_score(df[0], probability_l2))