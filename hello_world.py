from sklearn import tree
features = [[144, 0], [128, 0], [150, 1], [165, 1]] #weight & texture(0 == smooth, 1 == bumpy)
labels = [0,0,1,1]  # 0 == apple, 1 == orange
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[148, 0]]))
