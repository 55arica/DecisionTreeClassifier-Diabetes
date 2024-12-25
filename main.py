import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv('diabetes_dataset.csv')

df.head()

x = df.drop(columns=['Outcome'])
y = df['Outcome']

# -----------------------------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# -----------------------------------------------------------------------------------------------------------------------------------------

model = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 42)
model.fit(x_train, y_train)

# -----------------------------------------------------------------------------------------------------------------------------------------

predictions = model.predict(x_test)

model_accuracy = accuracy_score(y_test, predictions)
classification_results = classification_report(y_test, predictions)

# -----------------------------------------------------------------------------------------------------------------------------------------

print(f'Model Accuracy: {model_accuracy}')

print(f'Classification Report: {classification_results}')
