import pandas as pd

df = pd.read_csv('../data/svm_data.csv', header = None)

X=df.drop(0, axis=1)
Y=df[0]

from sklearn.svm import SVC
cls = SVC(C = 100000, kernel='linear', random_state=241) #Создаём экземпляр класса,
#реализующего SVM-алгоритм с поддержкой параметра С (см. ниже)
#C - задаёт величину слагаемого в функционале, штрафующего за слишком маленькие отступы
#при таком значении (100000) слагаемое практически не учитывается, а значит алгоритм работает
#с выборкой, как с линейно разделимой
#см. формулу [2*||W||/c] штрафного слагаемого (регуляризатора)
#kernel - тип ядра, используемого в алгоритме

cls.fit(X=X, y=Y) #обучим модель

print(cls.support_) #Выводим индексы опорных объектов (индексация начинается с 0)