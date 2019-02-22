import pandas as pd
df = pd.read_csv('../Downloads/titanic.csv', index_col = 'PassengerId')

delColumns = ['SibSp', 'Parch'] #Создать список
df = df.drop(delColumns, axis=1) #Удалить по списку
df = df.drop(df.columns[[2, 5, 7, 8]], axis=1) #При вызове от списка вернет список
#.columns[i] вернет метку i-ого столбца не считая index_col

df = df.dropna(axis=0, how='any') #Удалить все строки (axis=0), содержащие хотя бы 1 NaN
#how='all' чтобы удалить только строки, содержащие все NaN

trgCol = df['Survived'] #Сохраним ответы для тестововой выборки (целевую переменную)
df = df.drop('Survived', 1) #Удалить строку/столбец (axis = 0/1) по названию

df['Sex'].replace(['male', 'female'], [1, 0], inplace=True) #Замена всех элементов на другие
#Не работает, если заменять элементы на элементы другого рода по одному
#Т.е. нельзя сначала заменить 'male' на 1, а потом 'female' на 0
#inplace=True означает, что изменять нужно тот элемент, который вызвал функцию, а не 
#возвращать структуру данных

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=241) #Создаем классификатор (дерево) 
#и обучаем (.fit)
clf = clf.fit(df, trgCol) #df - тестовая выборка (признаки),
#а trgCol - ответы на тестовой выборке

print(df.columns) #Вывод названий столбцов (списка их названий)
importances = clf.feature_importances_ #Возврщает массив важности признаков 
#(по их порядку в датафрейме)
print(importances) #Важность - как сильно улучшился критарий качества от применения этого
#признака в вершинах дерев