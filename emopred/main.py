import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

emoji_data = pd.read_csv('emojiData.csv')

X = emoji_data.drop(columns='emoji')
y = emoji_data['emoji']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# prediction = model.predict([[32, 1]])

# score = accuracy_score(y_test, prediction)

dotData = tree.export_graphviz(decision_tree=model, out_file="emoji_result.dot", feature_names=["age", "gender"], class_names=model.classes_, label="all", rounded=True, filled=True)

joblib.dump(model, "emoji_predictor.joblib")