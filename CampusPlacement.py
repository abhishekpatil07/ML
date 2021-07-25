#Data preprocessing

import pandas as pd #data processing
import numpy as np #linear algebra
import matplotlib.pyplot as plt #plotting correlation graph

dataset = pd.read_csv('/content/Placement_Data_Full_Class.csv') #uploading csv file
dataset

dataset.info() #checking the information i.e,type and contents in the dataset 

#Abbrevation of column headings

#ssc_p - Secondary Education percentage- 10th Grade
#ssc_b - Board of Education- Central/ Others
#hsc_p - Higher Secondary Education percentage- 12th Grade
#hsc_b - Board of Education- Central/ Others
#hsc_s - Specialization in Higher Secondary Education
#degree_p - Degree Percentage
#degree_t - Under Graduation(Degree type)- Field of degree education
#workex - Work Experience
#etest_p - Employability test percentage ( conducted by college)
#mba_p - MBA percentage
#status - Status of placement- Placed/Not placed

dataset.describe() #gives the statistics of the data frame

#Fill the empty rows of data in the data frame
#Encoding the data frame

#Converting the catagorical(object) data columns to int/float type
dataset['salary'].fillna(0,inplace=True) #there are null rows in salary data column because that student was not placed hence slary will be 0

dataset['ssc_b'].unique() # gies the obeject unique to the column
dataset['gender'] = dataset['gender'].replace(["M"], 1) #replaces the data in a column to user specified value/string
dataset['gender'] = dataset['gender'].replace(["F"], 0)

dataset['hsc_b'].unique()
dataset['ssc_b'] = dataset['ssc_b'].replace(["Central"],1)
dataset['ssc_b'] = dataset['ssc_b'].replace(["Others"],0)

dataset['hsc_s'].unique()
dataset['hsc_s'] = dataset['hsc_s'].replace(["Arts"],0)
dataset['hsc_s'] = dataset['hsc_s'].replace(["Commerce"],1)
dataset['hsc_s'] = dataset['hsc_s'].replace(["Science"],2)

dataset['degree_t'].unique()
dataset['degree_t'] = dataset['degree_t'].replace(["Sci&Tech"],0)
dataset['degree_t'] = dataset['degree_t'].replace(["Comm&Mgmt"],1)
dataset['degree_t'] = dataset['degree_t'].replace(["Others"],2)

dataset['workex'].unique()
dataset['workex'] = dataset['workex'].replace(["Yes"],1)
dataset['workex'] = dataset['workex'].replace(["No"],0)

dataset['specialisation'].unique()
dataset['specialisation'] = dataset['specialisation'].replace(["Mkt&HR"],1)
dataset['specialisation'] = dataset['specialisation'].replace(["Mkt&Fin"],0)

dataset['status'].unique()
dataset['status'] = dataset['status'].replace(["Not Placed"],0)
dataset['status'] = dataset['status'].replace(["Placed"],1)

#dataset.head()
#dataset.info()

X = np.array(dataset.drop(['sl_no','status','salary'],1)) #features
y = dataset.iloc[:, -2].values #target
x
y

#Correlation of the data

dataset.corr()['status'] #Gives the correlation of target with other features

#Plotting to know how the data columns are distributed in the dataset
plt.plot(dataset['status'], dataset['ssc_p'],'r^')
plt.xlabel('Status')
plt.ylabel('Secondary Education percentage')

plt.plot(dataset['status'], dataset['hsc_p'],'y^')
plt.xlabel('Status')
plt.ylabel('Higher Secondary Education percentage')

plt.plot(dataset['status'], dataset['degree_p'],'m^')
plt.xlabel('Status')
plt.ylabel('Degree percentage')

#It is observed that features like ssc_p( Secondary Education percentage- 10th Grade), hsc_p(Higher Secondary Education percentage- 12th Grade), degree_p(Degree Percentage), workex(Work Experience) has more influence on getting placed.
#While other features are of less impoertance to get placed.

#Splitting dataset into trains and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

#Standardizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_test = y_test.reshape(len(y_test),1) #converting to a 2D array
y_train = y_train.reshape(len(y_train),1) #converting to a 2D array
#Our training and testing data is now ready to perform machine learning algorithm

#Training the model for different algorithms
#necessary tools from Scikit-learn library
from sklearn.tree import DecisionTreeRegressor #Decision tree algorithm
from sklearn.ensemble import RandomForestRegressor  #Random forest regression algorithm
from sklearn.svm import SVR  #Scalar vector machine regression algorithm
from sklearn.linear_model import LogisticRegression #Logistic regression algorithm

#necessary tools from Scikit-learn library
from sklearn.tree import DecisionTreeClassifier #Decision tree classification algorithm
from sklearn.ensemble import RandomForestClassifier #Random forest classification algorithm
from sklearn.svm import SVC #Scalar vector machine classification algorithm
from sklearn.neighbors import KNeighborsClassifier #K-neighbors classification algorithm

d_reg = DecisionTreeRegressor()
r_reg = RandomForestRegressor(n_estimators=500)
s_reg = SVR()
l_reg = LogisticRegression()

d_cla = DecisionTreeClassifier()
r_cla = RandomForestClassifier(n_estimators=500) 
s_cla = SVC()
k_cla = KNeighborsClassifier(n_neighbors=10)


d_reg.fit(X_train, y_train)
r_reg.fit(X_train, y_train)
s_reg.fit(X_train, y_train)
l_reg.fit(X_train, y_train)

d_cla.fit(X_train, y_train)
r_cla.fit(X_train, y_train)
s_cla.fit(X_train, y_train)
k_cla.fit(X_train, y_train)

#Predicting the result for trained data
d_pred = d_reg.predict(X_test)
r_pred = r_reg.predict(X_test)
s_pred = s_reg.predict(X_test)
l_pred = l_reg.predict(X_test)

dc_pred = d_cla.predict(X_test)
rc_pred = r_cla.predict(X_test)
kc_pred = k_cla.predict(X_test)
sc_pred = s_cla.predict(X_test)

from sklearn.metrics import r2_score #evaluates the performance of regression model
from sklearn.metrics import mean_squared_error #mean squared error of the regression model
from sklearn.metrics import accuracy_score #to find the accuray of classification algorithms 
from sklearn.metrics import confusion_matrix #to analyze the accuracy of classification algorithm


#r2_score
d = r2_score(y_test, d_pred)
r = r2_score(y_test, r_pred)
s = r2_score(y_test, s_pred)
l = r2_score(y_test, l_pred)

print('Decisiontree regression:',d)
print('Random Forest Regression:',r)
print('Scalar Vector Machine Regression:',s)
print('Logistic Regression:',l)


#mean square error
d_rmse = np.sqrt(mean_squared_error(y_test, d_pred))
r_rmse = np.sqrt(mean_squared_error(y_test, r_pred))
s_rmse = np.sqrt(mean_squared_error(y_test,s_pred))
l_rmse = np.sqrt(mean_squared_error(y_test, l_pred))

print('Decisiontree regression:',d_rmse)
print('Random Forest Regression:',r_rmse)
print('Scalar Vector Machine Regression:',s_rmse)
print('Logistic Regression:',l_rmse)


#accuracy
ac_dcla = accuracy_score(y_test, dc_pred)
ac_rcla = accuracy_score(y_test, rc_pred)
ac_scla = accuracy_score(y_test, sc_pred)
ac_kcla = accuracy_score(y_test, kc_pred)

print('Decisiontree Classificaton:',ac_dcla)
print('Random Forest Classificaton:',ac_rcla)
print('Scalar Vector Machine Classification:',ac_scla)
print('KNeighbors Classificaton:',ac_kcla)


#confusion Matrix
cm_dcla = confusion_matrix(y_test, dc_pred)
cm_rcla = confusion_matrix(y_test, rc_pred)
cm_scla = confusion_matrix(y_test, sc_pred)
cm_kcla = confusion_matrix(y_test, kc_pred)

print('Decisiontree Classificaton:\n',cm_dcla)
print('Random Forest Classificaton:\n',cm_rcla)
print('Scalar Vector Machine Classification:\n',cm_scla)
print('KNeighbors Classificaton:\n',cm_kcla)



#Conclusion
#Clearly Regression algorithm is not suitable for this data frame as it is not continuous data.

print('Algorithm with best accuracy/result is Scalar Vector Machine Classifier:',ac_scla*100,'%')
print('Algorithm with best accuracy/result is KNeighbors Classificaton::',ac_kcla*100,'%')
print('Algorithm with best accuracy/result is Random Forest Classificaton::',ac_rcla*100,'%')
print("Average result:",(ac_scla+ac_kcla+ac_rcla)/3)
 





