#Data Preprocessing
import pandas as pd # data processing
import numpy as np # linear algebra
import matplotlib.pyplot as plt #plotting correlation graph

dataset = pd.read_csv('/content/winequality-red.csv') #uploading csv file
dataset

dataset.info() #checking the information i.e,type and contents in the dataset 
#target is quality; rest of the colums are features

#Correlation of data
X = dataset.iloc[:, :-1].values #features
y = dataset.iloc[:, -1].values #target
x
y

#Here we see that fixed acidity does not give much specification to classify the quality.
plt.plot(dataset['quality'], dataset['fixed acidity'],'b+')
plt.xlabel('Quality')
plt.ylabel('fixed acidity')
#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality
plt.plot(dataset['quality'], dataset['volatile acidity'],'r^')
plt.xlabel('Quality')
plt.ylabel('volatile acidity')
#Composition of citric acid go higher as we go higher in the quality of the wine
plt.plot(dataset['quality'], dataset['citric acid'],'g^')
plt.xlabel('Quality')
plt.ylabel('citric acid')
#Here we see that residual sugar does not give any specification to classify the quality.
plt.plot(dataset['quality'], dataset['residual sugar'],'y+')
plt.xlabel('Quality')
plt.ylabel('residual sugar')
#Composition of chloride goes down as we go higher in the quality of the wine
plt.plot(dataset['quality'], dataset['chlorides'],'m^')
plt.xlabel('Quality')
plt.ylabel('chlorides')
#Here we see free sulfur dioxide does not give much specification to classify the quality.
plt.plot(dataset['quality'], dataset['free sulfur dioxide'],'r+')
plt.xlabel('Quality')
plt.ylabel('free sulfur dioxide')
#Here we see that total sulfur dioxide does not give any specification to classify the quality.
plt.plot(dataset['quality'], dataset['total sulfur dioxide'],'y+')
plt.xlabel('Quality')
plt.ylabel('total sulfur dioxide')
#Here we see that density does not give much specification to classify the quality.
plt.plot(dataset['quality'], dataset['density'],'g+')
plt.xlabel('Quality')
plt.ylabel('density')
#Here we see that pH does not give much specification to classify the quality.
plt.plot(dataset['quality'], dataset['pH'],'r+')
plt.xlabel('Quality')
plt.ylabel('pH')
#Here we see that sulphates does not give any specification to classify the quality.
plt.plot(dataset['quality'], dataset['sulphates'],'k+')
plt.xlabel('Quality')
plt.ylabel('sulphates')
#Alcohol level goes higher as te quality of wine increases
plt.plot(dataset['quality'], dataset['alcohol'],'c^')
plt.xlabel('Quality')
plt.ylabel('alcohol')

#Spliting the data into trains and tests to further analyze the data 
from sklearn.model_selection import train_test_split
X_train, X_test ,y_train, y_test = train_test_split(X,y, test_size=0.20)
len(X_test)
len(y_test)

##Standardizing the data
#Applying standard scaling for optimized output
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
t_test = y_test.reshape(len(y_test),1) #converting to 2D array
t_train = y_train.reshape(len(y_train),1) #converting to 2D array

#t_train
#t_test

X_sc = StandardScaler()
y_sc = StandardScaler()

X_std_train = X_sc.fit_transform(X_train)
y_std_train = y_sc.fit_transform(t_train)

#X_std_train
#y_std_train

X_std_test = X_sc.transform(X_test)
y_std_test = y_sc.transform(t_test)

#y_std_test
#X_std_test

#Our training and testing data is now ready to perform machine learning algorithm
#Training the model for different ML algorithms

#importing necessary tools from Scikit-learn library
from sklearn.linear_model import LinearRegression #Multiple linear regression algorithm 
from sklearn.preprocessing import PolynomialFeatures #Polynomial regression algorithm
from sklearn.ensemble import RandomForestRegressor #Random forest regression algorithm
from sklearn.tree import DecisionTreeRegressor #Decision tree algorithm
from sklearn.svm import SVR #Scalar vector machine regression algorithm
from sklearn.linear_model import LogisticRegression #Logistic regression algorithm

from sklearn.tree import DecisionTreeClassifier #Decision tree classification algorithm
from sklearn.ensemble import RandomForestClassifier #Random forest classification algorithm
from sklearn.neighbors import KNeighborsClassifier #K-neighbors classification algorithm
from sklearn.svm import SVC #Scalar vector machine classification algorithm

#regression
m_reg = LinearRegression()
p_reg = LinearRegression()
r_reg = RandomForestRegressor(n_estimators=500)
d_reg = DecisionTreeRegressor()
s_reg = SVR()
l_reg = LogisticRegression()

#classification
d_cla = DecisionTreeClassifier()
r_cla = RandomForestClassifier(n_estimators=500)
k_cla = KNeighborsClassifier(n_neighbors=10)
s_cla = SVC()
X_poly = PolynomialFeatures(degree = 2) #all polynomial combinations of features with degree 2
X_poly = X_poly.fit_transform(X_train)

m_reg.fit(X_train,y_train)
p_reg.fit(X_poly, y_train)
r_reg.fit(X_train,y_train)
d_reg.fit(X_train,y_train)
s_reg.fit(X_std_train, y_std_train)
l_reg.fit(X_train,y_train)

d_cla.fit(X_train, y_train)
r_cla.fit(X_train, y_train)
s_cla.fit(X_train, y_train)
k_cla.fit(X_train, y_train)
temp = PolynomialFeatures(degree = 2)
temp = temp.fit_transform(X_test)

#Predicting the result for the trained data

m_pred = m_reg.predict(X_test)
p_pred = p_reg.predict(temp)
d_pred = d_reg.predict(X_test)
r_pred = r_reg.predict(X_test)
s_pred = s_reg.predict(X_std_test)
l_pred = l_reg.predict(X_test)


dc_pred = d_cla.predict(X_test)
rc_pred = r_cla.predict(X_test)
kc_pred = k_cla.predict(X_test)
sc_pred = s_cla.predict(X_test)

#accuracy, error score and confusion matrix


from sklearn.metrics import r2_score #evaluates the performance of regression model
from sklearn.metrics import mean_squared_error #mean squared error of the regression model
from sklearn.metrics import accuracy_score #to find the accuray of classification algorithms 
from sklearn.metrics import confusion_matrix #to analyze the accuracy of classification algorithm

#R2 Score for all the regression algorithm
m = r2_score(y_test, m_pred)
p = r2_score(y_test, p_pred)
d = r2_score(y_test, d_pred)
r = r2_score(y_test, r_pred)
s = r2_score(y_std_test, s_pred)
l = r2_score(y_test, l_pred)


print('Linear Regression:',m)
print('Polynomial Regression:',p)
print('Decisiontree regression:',d)
print('Random Forest Regression:',r)
print('Scalar Vector Machine Regression:',s)
print('Logistic Regression:',l)

#Mean square error

m_rmse = np.sqrt(mean_squared_error(y_test, m_pred))
p_rmse = np.sqrt(mean_squared_error(y_test, p_pred))
d_rmse = np.sqrt(mean_squared_error(y_test, d_pred))
r_rmse = np.sqrt(mean_squared_error(y_test, r_pred))
s_rmse = np.sqrt(mean_squared_error(y_sc.inverse_transform(y_std_test), y_sc.inverse_transform(s_pred)))
l_rmse = np.sqrt(mean_squared_error(y_test, l_pred))

print('Linear Regression:',m_rmse)
print('Polynomial Regression:',p_rmse)
print('Decisiontree regression:',d_rmse)
print('Random Forest Regression:',r_rmse)
print('Scalar Vector Machine Regression:',s_rmse)
print('Logistic Regression:',l_rmse)

#Accuracy score
ac_dcla = accuracy_score(y_test, dc_pred)
ac_rcla = accuracy_score(y_test, rc_pred)
ac_scla = accuracy_score(y_test, sc_pred)
ac_kcla = accuracy_score(y_test, kc_pred)

print('Decisiontree Classificaton:',ac_dcla)
print('Random Forest Classificaton:',ac_rcla)
print('Scalar Vector Machine Classification:',ac_scla)
print('KNeighbors Classificaton:',ac_kcla)

#Confusion matrix
cm_dcla = confusion_matrix(y_test, dc_pred)
cm_rcla = confusion_matrix(y_test, rc_pred)
cm_scla = confusion_matrix(y_test, sc_pred)
cm_kcla = confusion_matrix(y_test, kc_pred)

print('Decisiontree Classificaton:\n',cm_dcla)
print('Random Forest Classificaton:\n',cm_rcla)
print('Scalar Vector Machine Classification:\n',cm_scla)
print('KNeighbors Classificaton:\n',cm_kcla)


#Sorting the quality of wine
dataset['quality']
reviews = []
for i in dataset['quality']:
    if i >= 4 and i <= 6:
        reviews.append('Bad')
    elif i >6 and i <= 9:
        reviews.append('Good')
reviews

#Conclusion
print('Algorithm with best accuracy/result is Randomforest Classifier:',ac_rcla*100,'%')
print('Algorithm with best accuracy/result is Randomforest Regression:',r*100,'%')



