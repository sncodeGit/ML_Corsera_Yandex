import numpy as np

from sklearn.datasets import load_boston
ds = load_boston() #sklearn.datasets - набор датасетов для загрузки
#Эти датасеты представляют из себя некоторые dict-like объекты

print(ds.keys()) #Получим ключи данного объекта
#DESCR - описание набора данных и некоторая статистика
#data - данные
#и т.д.

ds.data[:10] #Получим признаковые описания первых десяти объектов

from sklearn.preprocessing import scale
ds.data = scale(ds.data) #Масштабируем признаки

param = np.linspace(start=1, stop=10, num=200) #Получим массив из 200 значений параметра p
#С равным шагом из интервала [start,stop], включая границы

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
#См. KNN_KChoice

scores = [] #Список значений cv_score на разных итерациях
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
for i in range(0, 200):
    model = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski',
                                p=param[i]) 
    #Создадим саму модель
    #weights='distance' - задает использовать ли веса для элементов
    #их значение при данном задании зависит от расстояния до элемента
    #metric - задает используемую метрику
    #p - настраиваемый параметр в выбранной метрике (метрике Минковского)
    sv_score = cross_val_score(estimator=model, X=ds.data, y=ds.target, cv=kf,
                               scoring='neg_mean_squared_error')
    #neg_mean_squared_error - среднеквадратичная ошибка (метод оценивания качества)
    #Возвращает значения в интервале (-inf,0], что странно учитывая название метода
    #такое решение было принято для удобства максимизации результата в некоторых случаях (?)
    #При таком подходе 0 - идеальное значение для модели, с точки зрения правильности
    #выводимых ответов
    scores.append(sv_score.sum()/5)
    
print('The max value is reaches when p =', param[scores.index(max(scores))])