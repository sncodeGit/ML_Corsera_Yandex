import pandas as pd
df1 = pd.read_csv('../data/classification.csv')

#Подсчет величин TP, FP, FN, TN
#TP + FP + FN + TN = количество объектов (для хотя бы минимальной проверки)
TP = FP = TN = FN = 0
for i in range(0, len(df1)):
    if (df1['true'][i] == df1['pred'][i]):
        if (df1['true'][i] == 1): 
            TP += 1
        else:
            TN += 1
    else:
        if (df1['true'][i] == 1):
            FN += 1
        else: 
            FP += 1

print('Matrix:')            
print(TP, FP, FN, TN)

Y_true = df1['true']
Y_pred = df1['pred']

import sklearn.metrics as skm
print('\nQuality metrics (binary classification):')
#Accuracy
print(skm.accuracy_score(Y_true, Y_pred), (TP + TN) / (TP + TN + FP + FN))
#Precision
precision = TP / (TP + FP)
print(skm.precision_score(Y_true, Y_pred), precision)
#Recall
recall = TP / (TP + FN)
print(skm.recall_score(Y_true, Y_pred), recall)
#F-мера
print(skm.f1_score(Y_true, Y_pred), 2 * precision * recall / (precision + recall))

#Новые данные для методов с результатом в виде 
df2 = pd.read_csv('../data/scores.csv')

Y_true = df2['true']
Y_logreg = df2['score_logreg']
Y_svm = df2['score_svm']
Y_knn = df2['score_knn']
Y_tree = df2['score_tree']

#AUC-ROC для разных методов
print('\nAUC-ROC:')
print(skm.roc_auc_score(Y_true, Y_logreg), skm.roc_auc_score(Y_true, Y_svm))
print(skm.roc_auc_score(Y_true, Y_svm), skm.roc_auc_score(Y_true, Y_tree))

#Поиск максимального значения precision при recall>=0.7
#На вход подаются правильные ответы и вероятностная оценка классификатора
def max_prec(Y_method, Y_true = Y_true):
    prec, rec, thresholds = skm.precision_recall_curve(Y_true, Y_method)
    #Возвращает массив значений precision и recall для некоторых точек PRC
    #thresholds задаёт пороговые значения вероятности для вычисления precision и recall
    points_count = len(prec)
    max_prec = 0
    for i in range(0, points_count):
        if (rec[i] >= 0.7 and prec[i] > max_prec):
            max_prec = prec[i]
    return max_prec

print('\nMax precisions (recall >= 0.7):')
print(max_prec(Y_logreg))
print(max_prec(Y_svm))
print(max_prec(Y_knn))
print(max_prec(Y_tree))