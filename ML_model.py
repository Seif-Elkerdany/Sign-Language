#modules
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

#reading the data
df = pd.read_csv(r"C:\Users\Saif Elkerdany\MyTraining_Data.csv",header = None)

#pre-procssing
X = df.drop(columns= [42])
Y = df[42]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,shuffle=True)

#model
clf = MLPClassifier(random_state=2, max_iter=1000,activation = "tanh" , alpha = 0.5).fit(x_train, y_train)
y_predict = clf.predict(x_test)

#acurracy
score = accuracy_score(y_predict, y_test)
print(f"Accuracy: {score * 100} %")

#exporting the model
with open("Model_mydata.pkl", "wb") as f:
    pickle.dump(clf, f)