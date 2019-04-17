from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 3 different machine learning techniques from the sklearn library tackle the Iris flower identifying
# challenge, each achieving extremely high rates of probablity, reaching the optimum of 1 in some examples.

iris = load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=.2, random_state=42)


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction1 = clf.predict(X_test)
print(accuracy_score(y_test, prediction1))

clf2 = SVC()
clf2.fit(X_train, y_train)
prediction2 = clf2.predict(X_test)
print(accuracy_score(y_test, prediction2))

clf3 = RandomForestClassifier(n_estimators=1000, max_depth=2)
clf3.fit(X_train, y_train)
prediction3 = clf3.predict(X_test)
print(accuracy_score(y_test, prediction3))
