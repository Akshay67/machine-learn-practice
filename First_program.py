from sklearn import tree

X = [[181, 80, 44], [212, 95, 69], [195, 68, 58]]

Y = ['male','female','female']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)

prediction = clf.predict([[212, 95, 69]])

print(prediction)