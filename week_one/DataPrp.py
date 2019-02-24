import pandas as pd
df = pd.read_csv('../data/titanic.csv', #Абсолютный или относительный путь до файла
                index_col = 'PassengerId') #Колонка PassangerId задаёт нумерацию 
                #строк этого датафрейма

print(len(df)) #Общее количество строк

print(len(df[df['Sex'] == 'male'])) #Количество выбранных по данному признаку строк
print(df['Sex'].value_counts()) #Вывести статистику по этому столбцу

print(df['Survived'].value_counts())

print(df['Pclass'].value_counts())

numNaN = df['Age'].isnull().sum() #Количество пропусков в данном столбце
sum = df['Age'].sum() #Cумма элементов
num = len(df['Age']) - numNaN #Количество непустых элементов в столбце
print(sum / num) #Среднее без функции mean()
print(df['Age'].mean()) #Среднее, исключая пустые элементы
print(df['Age'].median()) #Медиана

print(df['SibSp'].corr(df['Parch'])) #Корреляция Пирсона method='pearson'

#Поиск самого частого женского имени
