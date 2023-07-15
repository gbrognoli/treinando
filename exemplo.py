import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv("data.csv")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, data["target"], test_size=0.25)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)

print("Accuracy:", score)
