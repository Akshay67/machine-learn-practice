# Load the libraries
import pandas
from pandas import scatter_matrix
import matplotlib.pyplot as pt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) # THis line read the dataset in cvs format

print(dataset.shape) # This will print the number of rows and column's in the iris dataset

print(dataset.head(30)) # This will print first 30 results

print(dataset.describe()) # This will print the data in detail


