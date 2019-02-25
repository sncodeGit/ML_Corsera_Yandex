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

diction = dict() #Cловарь, в котором будут храниться имена (ключи) и количество их вхождений
names = df[df['Sex'] == 'female']['Name'].get_values() #Массив всех женских имен датафрейма
#Выбираем сначала только объекты с 'female' в поле 'Sex', а затем - их имена
#get.values() - получить список значений DataFraim (двумерный) или Series (одномерный)

for name in names: #Нахождение самого first name из строки с полным именем
    left_brack_pos = name.find('(') #Поиск первого вхождения подстроки (-1, если не найдено)
    
    #Перепишем name так, чтобы first name стояло в начале строки, содержащейся в name
    #Если есть скобки, то first name - это первое слово в них
    #Если скобок нет, то first name - это первое слово после Miss. (Mrs. и т.п.)
    if (left_brack_pos != -1):
        #Удалим всё до скобки (включительно)
        name = name[left_brack_pos + 1:]
    else:
        #Удалим всю строку до точки (включительно) и следующий пробел
        name = name[name.find('.') + 2:] 
    
    space_pos = name.find(' ')
    if (space_pos != -1): #Если имя не стоит в конце строки (без пробела после)
        name = name[:space_pos]
    
    #Изменим состояние словаря в соответствии с полученным first name
    if (diction.get(name) == None): #get() возвращает значение ключа 
    #(None или другое значение по умолчанию, если ключ не найден)
        diction[name] = 1 #Создаем новую запись словаря
    else:
        diction[name] += 1 #Или инкрементируем счётчик вхождения данного имени

print(diction.items()) #Возвращает пары ключ-значение словаря
