import pandas as pd

train_df = pd.read_csv('../data/perceptron_train.csv', header=None)
test_df = pd.read_csv('../data/perceptron_test.csv', header=None)

from sklearn.linear_model import Perceptron
cls = Perceptron(random_state=241) #Создаём сам перцептрон
#Перцептрон = {S_elem, A_elem, R_elem}. 
#S_elem - рецепторы (сенсоры), находящиеся либо в состоянии покоя (0),
#либо в состоянии возбуждения (+1) при превышении входным сигналом некоторого порога
#могут активизироваться от выполния некоторого условия для входных данных
#A_elem - ассоциативные элементы, принимающие сигнал от некоторого набора (ассоциации) S_elem
#при превышении алгебраической суммы ассоциаций некоторой пороговой величины
#элемент активизируется (+1), иначе он находится в состоянии покоя (0)
#R_elem - реагирующие (действующие) элементы, действует как и A_elem, но в качестве 
#ассоциации выступает набор A_elem, каждый элемент которого связнан с некоторым весом Wi
#при положительном значении линейной суммы ассоциации (суммы с учётом Wi), выдаёт сигнал +1
#при отрицательном - -1, при нуле - 0 (либо неопределённый сигнал)

cls.fit(train_df.drop(0, axis=1), train_df[0]) #Обучаем классификатор
predictions = cls.predict(test_df.drop(0, axis=1)) #Предсказания обученной модели по признакам

from sklearn.metrics import accuracy_score
print('Originally : ', accuracy_score(y_true=test_df[0], y_pred=predictions, 
                                      normalize=True))
#Доля правильных ответов для ненормализованных данных
#y_true - правильные ответы, y_pred - предсказания модели
#normalize=True (default) - доля правильных ответов, а не их количество (False)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #Создаём экземпляр служебного класса для предобработки данных
scaler.fit(train_df.drop(0, axis=1)) #Вычисляем среднее и дисперсию каждого признака, которые
#будут использованы в дальнейших преобразованиях
X_train_scaled = scaler.transform(X=train_df.drop(0, axis=1), y=train_df[0]) #Нормализуем
#признаки обучающей выборки по формуле [(feature - feature_mean) / feature_dispersion]
#Также для объединения fit() и transform() можно использовать fit_transform()
X_test_scaled = scaler.transform(X=test_df.drop(0, axis=1), y=test_df[0])#Нормализуем
#признаки контрольной выборки средним и дисперсией обучающей выборки
#т.к. мы же использовали эту схему предобработки данных для обучения модели

cls.fit(X_train_scaled, train_df[0]) #Переобучаем классификатор и
predictions = cls.predict(X_test_scaled) #предсказываем новые ответы
print('After normalization :', accuracy_score(y_true=test_df[0], y_pred=predictions,
                                              normalize=True))