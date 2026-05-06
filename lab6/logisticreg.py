import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error , accuracy_score, confusion_matrix

df= pd.read_csv('Housing_Data.csv')


features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated']


X = df[features]
y = (df['price'] > 100000).astype(int)
df['price_label'] = y

print(df.head())

clf = LogisticRegression(max_iter=10000)

X_train , X_test , y_train , y_test = train_test_split(
    X , y ,test_size=0.2 , random_state=42
)
clf.fit(X_train , y_train)


predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(accuracy , cm )