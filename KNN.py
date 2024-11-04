import numpy as np
import pandas as pd
data = pd.read_csv('/content/Iris.csv')
data.head()
data.info()
data['Species'].value_counts()
x = data.iloc[:,1:5]
x.head()
y = data.iloc[:,-1]
y.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
x[0:5]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.shape
x_test.shape

from sklearn.neighbors import KNeighborsClassifier
correct_sum = []
for i in range(1,20):
  model = KNeighborsClassifier(n_neighbors=i)
  model.fit(x_train,y_train)
  pred = model.predict(x_test)
  correct  = np.sum(pred == y_test)
  correct_sum.append(correct)

result = pd.DataFrame(data = correct_sum)
result.index = result.index + 1
result.T

model = KNeighborsClassifier(n_neighbors=13)
model.fit(x_train,y_train)
pred = model.predict(x_test)
pred[0:5]

y_test[0:5]

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)
