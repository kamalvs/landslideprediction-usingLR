#Basic Important Modules

import pandas as pd
import seaborn as sns
import pickle
sns.set(style='whitegrid', color_codes=True)
import warnings
warnings.filterwarnings('ignore')

#Sckit Learn Specific Modules

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


#ML Model

from sklearn.linear_model import LogisticRegression

#importing dataset
df = pd.read_csv(r'C:\Users\Kamal\Desktop\project\LS_PREDICTION_LR\Dis2.csv')

#splitting x & y
X = df[['CHANGE','ELEVATION','SLOPE','TWI','DRAINAGE','NDVI','RAINFALL','GEOLOGY']]
y = df[['Target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15, stratify= y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)


def tune_model(classifier, param_grid, X, y ):
  grid = GridSearchCV(classifier, param_grid, refit = True, cv =cv, verbose = 3, n_jobs=4)
  grid = grid.fit(X, y)
  return grid

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'penalty':['l1', 'l2', 'elasticnet']}

lR = tune_model(LogisticRegression(), param_grid, X_train, y_train)

#Fitting Logistic Regression to the training set
lR = LogisticRegression(penalty='l2', solver= 'newton-cg', C= 0.01).fit(X_train, y_train)
lR.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(lR, open("model.pkl", "wb"))

