from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

#load data
data = pd.read_csv(r'housing.csv')

#clean data
data = data.dropna()

#encode categorical features
data = data.join(pd.get_dummies(data['ocean_proximity']))
data.drop('ocean_proximity', inplace=True, axis=1)
data.drop('ISLAND', inplace=True, axis=1)

#split data
y = pd.DataFrame(data, columns= ['median_house_value'])
X = data.drop('median_house_value', 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#train model
lReg = LinearRegression()
model = lReg.fit(X_train, y_train)

#save model
joblib.dump(model, 'model.sav')

#predict test set
y_pred = model.predict(X_test)

#check accuracy of the model
print(model.score(X_test, y_test))
