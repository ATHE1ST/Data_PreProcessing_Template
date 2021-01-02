
#1. Kutuphaneler
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#2. Veri onisleme
#2.1. Veri Yukleme
veriler = pd.read_csv("eksikveriler.csv")

#test
print(veriler)

#veri on isleme

boy = veriler[["boy"]]
print(boy)


boykilo = veriler[["boy","kilo"]]
print(boykilo)


#eksik veriler
#sci - kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = veriler.iloc[:,1:4].values
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)

#Encoder: Kategorik Veriler -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

#Laber Encoder çizgisel olarak nümerik yapar
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)




#One Hot Encoder dönüştürmek istenilen değerleri kolon başlığı yapar ve hangisi varsa ona 1 diğerlerine 0 değerini verir
ohe = preprocessing.OneHotEncoder()  
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
 

#Numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ("fr","tr","us"))
print(sonuc)

sonuc2 = pd.DataFrame(data = yas, index = range(22), columns = ("boy","kilo","yas"))
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])


#Dataframe birlestirme islemi
s = pd.concat([sonuc,sonuc2], axis = 1)
print(s)

s2 = pd.concat([s,sonuc3], axis = 1)
print(s2)

#Verilerin test ve train icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)


#Verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)















































