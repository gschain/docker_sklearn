

from sklearn.externals import joblib


model = joblib.load('./model.m')
print(model.predict([[5,5,3],[2,2,2]]))

