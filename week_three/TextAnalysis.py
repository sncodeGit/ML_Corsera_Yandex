from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(subset = 'all', 
                                         categories = ['alt.atheism', 'sci.space']) 
#Загрузить датасет новостей
#subset = '' - какие данные загружать: тестовые, тренировочные или все
#categories = '' - выбор интересующих категорий новостей

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer() #Вспомогательный класс для преобразования коллекции
#текстов в матрицу* TF-IDF признаков
#*матрицу: выглядит как массив [(i, j) value], а преобразовать в обычную двумерную матрицу
#вида {matrix(i,j)} i = 1..n, j = 1..m можно методом toarray()
#?скорее всего это read-only матрица
X_vectorized = vectorizer.fit_transform(newsgroups.data) 
#Найдем набор слов, входящих в коллекцию текстов (vocabulary) и их IDF [fit]
#А также преобразуем эту коллекцию в матрицу TF-IDF признаков на основе вычисленных
#vocabulary и IDF [transform]

from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42) #Создадим генератор разбиений

from sklearn.svm import SVC
search_clf = SVC(kernel='linear', random_state=241) #Создадим сам классификатор

import numpy
from sklearn.model_selection import GridSearchCV
grid = {'C': numpy.power(10.0, numpy.arange(-5, 6))} #Получим словарь {'C': [10^-5...10^5]}
#power() - поэлементное возведение в степень элементов первого массива,
#где степень задаёт второй массив
#arange() - возвращает массив равномерно распределённых значений (шаг по умолчанию = 1)
#(от первого эл-та до второй-шаг)
gs = GridSearchCV(estimator=search_clf, param_grid=grid, scoring='accuracy', cv=cv)
#param_grid - словарь с именами параметров в качестве ключей и списками значений этого
#параметра для перебора в качестве значения по ключу
gs.fit(X_vectorized, newsgroups.target) #Производим подбор параметра
best_c = gs.best_params_['C'] #best_params_ = {'C': 1.0} (словарь)

final_clf = SVC(C=best_c, kernel='linear', random_state=241) 
final_clf.fit(X_vectorized, newsgroups.target)
#Можно было использовать атрибут GridSearchCV best_estimator_,
#который вернул бы лучшую модель

coef = final_clf.coef_.toarray() #Преобразуем coef_ (read-only матрица) в двумерный массив
coef = coef[0] #Если количество классов равно двум, то в массив будет в сущности одномерный
higher_ten_ind = numpy.argsort(abs(coef))[-10:] #Получим индексы 10-и наибольших элементов
#argsort() - сортирует переданный массив в порядке возрастания, а возвращает массив 
#индексов элементов изначального массива Ex: [10, 5] --> [1, 0]

feature_mapping = vectorizer.get_feature_names() #Получим имена признаков, индексы
#которых совпадают с индексами весов в .coef_
for i in higher_ten_ind: #Выведем слова (т.е. в этом случае имена_признаков == слова)
    print(feature_mapping[i])