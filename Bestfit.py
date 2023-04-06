import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data 

CSV_Data = pd.read_csv("D:/Semester 6/Machine Learning/Assignment 1/canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]

#Convert in 2 Rank

X = np.array(X)
Y = np.array(Y)
X = X[:,np.newaxis]
Y = Y[:,np.newaxis]

#Scaling_Of_X

min_x = X.min()
max_x = X.max()
scale_x_numerator = X-min_x
scale_x_denominator = max_x-min_x 

scale_x1 = scale_x_numerator/scale_x_denominator
scale_x2 = np.power(scale_x1[:,0],2)
scale_x3 = np.power(scale_x1[:,0],3)
scale_x4 = np.power(scale_x1[:,0],4)
scale_x5 = np.power(scale_x1[:,0],5)
scale_x6 = np.power(scale_x1[:,0],6)
scale_x7 = np.power(scale_x1[:,0],7)
scale_x8 = np.power(scale_x1[:,0],8)
scale_x9 = np.power(scale_x1[:,0],9)
scale_x10 = np.power(scale_x1[:,0],10)
scale_x11 = np.power(scale_x1[:,0],11)
scale_x12 = np.power(scale_x1[:,0],12)
scale_x13 = np.power(scale_x1[:,0],13)
scale_x14 = np.power(scale_x1[:,0],14)
scale_x15 = np.power(scale_x1[:,0],15)
scale_x16 = np.power(scale_x1[:,0],16)
scale_x17 = np.power(scale_x1[:,0],17)
scale_x18 = np.power(scale_x1[:,0],18)
scale_x19 = np.power(scale_x1[:,0],19)
scale_x20 = np.power(scale_x1[:,0],20)
scale_x21 = np.power(scale_x1[:,0],21)
scale_x22 = np.power(scale_x1[:,0],22)
scale_x23 = np.power(scale_x1[:,0],23)
scale_x24 = np.power(scale_x1[:,0],24)
scale_x25 = np.power(scale_x1[:,0],25)
scale_x26 = np.power(scale_x1[:,0],26)
scale_x27 = np.power(scale_x1[:,0],27)
scale_x28 = np.power(scale_x1[:,0],28)
scale_x29 = np.power(scale_x1[:,0],29)
scale_x30 = np.power(scale_x1[:,0],30)

def making_it_2_rank(scale_x):
    scale_x = scale_x[:,np.newaxis]
    return scale_x

scale_x2 = making_it_2_rank(scale_x2)
scale_x3 = making_it_2_rank(scale_x3)
scale_x4 = making_it_2_rank(scale_x4)
scale_x5 = making_it_2_rank(scale_x5)
scale_x6 = making_it_2_rank(scale_x6)
scale_x7 = making_it_2_rank(scale_x7)
scale_x8 = making_it_2_rank(scale_x8)
scale_x9 = making_it_2_rank(scale_x9)
scale_x10 = making_it_2_rank(scale_x10)
scale_x11 = making_it_2_rank(scale_x11)
scale_x12 = making_it_2_rank(scale_x12)
scale_x13 = making_it_2_rank(scale_x13)
scale_x14 = making_it_2_rank(scale_x14)
scale_x15 = making_it_2_rank(scale_x15)
scale_x16 = making_it_2_rank(scale_x16)
scale_x17 = making_it_2_rank(scale_x17)
scale_x18 = making_it_2_rank(scale_x18)
scale_x19 = making_it_2_rank(scale_x19)
scale_x20 = making_it_2_rank(scale_x20)
scale_x21 = making_it_2_rank(scale_x21)
scale_x22 = making_it_2_rank(scale_x22)
scale_x23 = making_it_2_rank(scale_x23)
scale_x24 = making_it_2_rank(scale_x24)
scale_x25 = making_it_2_rank(scale_x25)
scale_x26 = making_it_2_rank(scale_x26)
scale_x27 = making_it_2_rank(scale_x27)
scale_x28 = making_it_2_rank(scale_x28)
scale_x29 = making_it_2_rank(scale_x29)
scale_x30 = making_it_2_rank(scale_x30)

# Having Extra Column With Values 1 for theta 0

m,col = scale_x1.shape
ones = np.ones((m,1))
theta = np.zeros((31,1))
iterations = 250000
alpha = 0.01

#Concatenation

X_scaling = np.concatenate((ones,scale_x1,scale_x2,scale_x3,scale_x4,scale_x5,scale_x6,scale_x7,scale_x8,scale_x9,scale_x10,scale_x11,scale_x12,scale_x13,scale_x14,scale_x15,scale_x16,scale_x17,scale_x18,scale_x19,scale_x20,scale_x21,scale_x22,scale_x23,scale_x24,scale_x25,scale_x26,scale_x27,scale_x28,scale_x29,scale_x30),axis=1)

#Scaling_Of_Y

min_y = Y.min()
max_y = Y.max()
scale_y_numerator = Y-min_y
scale_y_denominator = max_y-min_y
scale_y = scale_y_numerator/scale_y_denominator

def Get_cost_J(X,Y,Theta,m,lamda):
    predictions = np.dot(X,Theta)
    error = predictions-Y
    sqr_error = np.power(error, 2)
    sum_sqr_error = np.sum(sqr_error)
    regularization_term = lamda*(np.sum(np.power(Theta[1:],2)))
    cost = (sum_sqr_error + regularization_term)/(2*m)
    return cost

def gradient_decent(X, y, theta, alpha, iterations, lamda, m):
    history = []
    for i in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = (X.T.dot(error) + lamda * theta) / m
        gradient[0] = gradient[0] - lamda * theta[0] / m
        theta_new = theta - alpha * gradient
        theta = theta_new.copy()
        cost = Get_cost_J(X, y, theta, m, lamda)
        history.append(float(cost))
    return theta, history

Theta, hist = gradient_decent(X_scaling, scale_y, theta, alpha, iterations,1,m)
 
final_prediction = np.dot(X_scaling,Theta)

plt.scatter(X_scaling[:,1],scale_y)
plt.plot(X_scaling[:,1],final_prediction,color="red")
plt.show()









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Reading and visualizing data 
CSV_Data = pd.read_csv("D:/Semester 6/Machine Learning/Assignment 1/canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]

#Convert in 2 Rank
X = np.array(X)
Y = np.array(Y)
X = X[:,np.newaxis]
Y = Y[:,np.newaxis]

# Create Linear Regression object
model = LinearRegression()

# Train the model using the training sets
model.fit(X,Y)

# Make predictions using the testing set
Y_pred = model.predict(X)

# Plot the regression line
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

# Print the intercept and coefficient
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Reading and visualizing data 
CSV_Data = pd.read_csv("D:/Semester 6/Machine Learning/Assignment 1/canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]

# Convert data to 2D arrays
X = np.array(X).reshape(-1, 1)
Y = np.array(Y).reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30, random_state=42)

# Create Linear Regression object
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train,Y_train)

# Make predictions using the testing set
Y_pred = model.predict(X_test)

# Plot the regression line on the training data
plt.scatter(X_train, Y_train)
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Training Set')
plt.show()

# Plot the regression line on the testing data
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred, color='red')
plt.title('Testing Set')
plt.show()

# Print the intercept and coefficient
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)







#Assignment No 2
#Hafiz Muhammad Danish
#Phdcsf21m506

#Excercise No 2 part 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 #Reading Files
data = pd.read_csv('D:/Phd/First Semester/ML/Assignments/Assignemnt_2/ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
data.head()
m = len(X)

y = y[:,np.newaxis]
mask = y == 1

adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

cX=X
cY=y
ones = np.ones((m,1))
X = np.hstack((ones,X))


# increasing number of features 

new1 = np.power(X[:,1],2)
new1= new1[:,np.newaxis]
X= np.hstack((X,new1))

new2 = np.power(X[:,2],2)
new2= new2[:,np.newaxis]
X= np.hstack((X,new2))

new3 = np.multiply(X[:,1],X[:,2])
new3= new3[:,np.newaxis]
X= np.hstack((X,new3))






theta = np.zeros((6,1))
iterations = 2500
alpha = 0.01





# sigmaoid fucntion

def sigmoid(h):
    g = 1/(1+np.exp(-h))
    return g
    


# Cost Fucction 

def Get_cost_J(X,Y,Theta,m,lamda):
    
    temp1 = np.multiply(Y,np.log(sigmoid(np.dot(X,Theta))))
    temp2 = np.multiply((1-Y),np.log(1-sigmoid(np.dot(X,Theta))))
    J  =(-1/m)*np.sum(temp1+temp2)
    reg = (lamda/2*m)*np.sum(np.power(theta,2))
    J = J+reg
    return J
# Grdient Decent

def gradient_decent(x,y,theta,alpha,iterations,m, lamda):
    history = np.zeros((iterations,1))
    for i in range(iterations):
        z = np.dot(x,theta)
        g = sigmoid(z)
        g = g-y
        theta = theta - alpha*((np.dot(x.T,g)*((1/m)) + (lamda/m*theta)))
        
        history[i] = Get_cost_J(x, y, theta, m,lamda)
    
    return (theta,history)


theta,hist = gradient_decent(X, y, theta, alpha, iterations, m,1) 
plt.plot(hist)
plt.show()




def getpridiction(X,theta):
    z= np.dot(theta.T,X)
    g = 1/(1+np.exp(-z))
    return g

K = np.array([[1],[80],[80]])
value=getpridiction(K, theta[0:3,0])
if value>0.5:
    print('Admittd')
else:
    print('Not admitted')
    

mask = y == 1
theta = theta[0:3,0]
theta = theta[:,np.newaxis]
# we got theta values -59.196165 0.466 0.462 put this in formula 0x+0x1+0x2 = 0
# then simply x2 = -x101/020 - 0/02
adm = plt.scatter(cX[mask][0].values, cX[mask][1].values)
not_adm = plt.scatter(cX[~mask][0].values, cX[~mask][1].values)
plt.plot(cX.iloc[:,0],-(theta[0,0]/theta[2,0])-(cX.iloc[:,0]*theta[1,0]/theta[2,0]))
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()
    
    