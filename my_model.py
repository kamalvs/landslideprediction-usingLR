# Data Pre-procesing Step
# importing libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# importing datasets
df = pd.read_csv("Dis2.csv")

#select target
x = df[["PROFILE","PLAN","CHANGE","LANDUSE","ELEVATION","SLOPE","ASPECT","TWI","SPI","DRAINAGE","NDVI","RAINFALL","FAULTLINES","ROAD","GEOLOGY"]]
y = df["Target"]


# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.33, random_state=15)

#feature Scaling
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

#Fitting Logistic Regression to the training set
classifier= LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
