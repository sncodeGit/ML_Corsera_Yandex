import pandas as pd

df = pd.read_csv('../data/wine.data', header=None) #Считываем файл как датафрейм
#header=None - нет заголовков для столбцов 

clsDF = df[0] #Метки класса (первый столбец)
featuresDF = df.drop(0, axis=1) #Признаки (все столбцы помимо первого с метками класса)

cls = clsDF.to_numpy() #Привести Series к массиву numpy
features = featuresDF.to_numpy() #Привести DataFrame к массиву numpy

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42) #Создает генератор разбиений для 
#тестирования и валидации
#n_splits - количество разбиений (блоков) при кросс-валидации
#shuffle - перемешать данные в том случае, если они заданы не в случайном порядке
#Другим способом генерации разбиений является train_test_split(features, cls)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
for k in range(1, 51): #От одного соседа до 50
    classifier = KNeighborsClassifier(n_neighbors=k) #Создаем классификатор
    #n_neighbors - количество соседей для метода KNN
    cv_score = cross_val_score(estimator=classifier, X=features, y=cls, cv=kf, 
                               scoring='accuracy')
    #Проводим оценку алгоритма методом кросс-валидации
    #estimator - оцениваемый алгоритм
    #X - массив признаков (неразделенный на test/train)
    #y - массив меток класса (ответов) для X
    #cv - количество блоков для кросс-валидации (fold) или генератор разбиений 
    #(в этом случе он)
    #scoring - используемый метод оценивания (accuracy - по количеству верных ответов)
    print(k, cv_score.sum()/5) #Так как метод выводит массив из 
    #оценок по каждому из fold-ов
    
from sklearn.preprocessing import scale
features = scale(features) #Масштабируем признаки (- среднее и / на стандартное отклонение)

scores = [] #Список значений cv_score на разных итерациях
print('After preprocessing:')
for k in range(1, 51): 
    classifier = KNeighborsClassifier(n_neighbors=k) 
    cv_score = cross_val_score(estimator=classifier, X=features, y=cls, cv=kf,
                               scoring='accuracy')
    scores.append(cv_score.sum()/5) #Добавление элемнта в конец списка

print('Max element: ', scores.index(max(scores))) #Вывод индекса максимального эл-та списка
#Его значение мы находим через функцию max(list_name)
