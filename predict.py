from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

class SklearnPredict:

    def __init__(self, X_train, Y_train, X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def logisticRegression(self):
        logreg = LogisticRegression()
        logreg.fit(self.X_train, self.Y_train)
        predict = logreg.predict(self.X_test)
        accuracy = round(logreg.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def svc(self):
        svc = SVC()
        svc.fit(self.X_train, self.Y_train)
        predict = svc.predict(self.X_test)
        accuracy = round(svc.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def knn(self, n_neighbors=0):
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(self.X_train, self.Y_train)
        predict = knn.predict(self.X_test)
        accuracy = round(knn.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def gaussian(self):
        gaussian = GaussianNB()
        gaussian.fit(self.X_train, self.Y_train)
        predict = gaussian.predict(self.X_test)
        accuracy = round(gaussian.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def perceptron(self):
        perceptron = Perceptron()
        perceptron.fit(self.X_train, self.Y_train)
        predict = perceptron.predict(self.X_test)
        accuracy = round(perceptron.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def linearSvc(self):
        linear_svc = LinearSVC()
        linear_svc.fit(self.X_train, self.Y_train)
        predict = linear_svc.predict(self.X_test)
        accuracy = round(linear_svc.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def sgd(self):
        sgd = SGDClassifier()
        sgd.fit(self.X_train, self.Y_train)
        predict = sgd.predict(self.X_test)
        accuracy = round(sgd.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def decisionTree(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.X_train, self.Y_train)
        predict = decision_tree.predict(self.X_test)
        accuracy = round(decision_tree.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict

    def randomForest(self):
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(self.X_train, self.Y_train)
        predict = random_forest.predict(self.X_test)
        random_forest.score(self.X_train, self.Y_train)
        accuracy = round(random_forest.score(self.X_train, self.Y_train) * 100, 2)
        return accuracy, predict


