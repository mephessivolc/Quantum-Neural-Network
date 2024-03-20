from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()

X = data.data 
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(X_train)
print(y_train)