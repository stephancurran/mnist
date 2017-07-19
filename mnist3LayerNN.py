
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('mnist.csv', sep=',', header=None)
X_train, X_test, y_train, y_test = train_test_split(data.ix[:, 1:], data.ix[:, 0], test_size=0.2, random_state=42)

startTime = time.time()
clf = MLPClassifier(max_iter=100, alpha=0.01, hidden_layer_sizes=(800,))
clf.fit(X_train, y_train)
print("Training time: {:f} minutes.".format((time.time() - startTime) / 60))

startTime = time.time()
score = clf.score(X_test, y_test)
print("Testing time: {:f} seconds.".format(time.time() - startTime))
print("Accuracy: {:f}%".format(score * 100))
