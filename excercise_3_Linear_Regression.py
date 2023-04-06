import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot

CSV_Data = pd.read_csv("D:/Semester 6/Machine Learning/Assignment 1/CarPrice_Assignment.csv")
CSV_Data = CSV_Data.replace(np.NaN,0)

#print(CSV_Data.shape)

#Separating Features For Training

X1 = CSV_Data.iloc[0:155,0:1]
X2 = CSV_Data.iloc[0:155,1:2]
X3 = CSV_Data.iloc[0:155,2:3]
X4 = CSV_Data.iloc[0:155,3:4]
X5 = CSV_Data.iloc[0:155,4:5]
X6 = CSV_Data.iloc[0:155,5:6]
Y = CSV_Data.iloc[0:155,-1]

#Label 

Y = np.array(Y)
Y = Y[:,np.newaxis]

#Scaling_Of_X_Features

def scaling_of_X_features(X):
    min_of_x = X.min()
    max_of_x = X.max()
    numerator = X-min_of_x
    denominator = max_of_x-min_of_x
    scaling = numerator/denominator
    return scaling

scale_x1 = scaling_of_X_features(X1)
scale_x2 = scaling_of_X_features(X2)
scale_x3 = scaling_of_X_features(X3)
scale_x4 = scaling_of_X_features(X4)
scale_x5 = scaling_of_X_features(X5)
scale_x6 = scaling_of_X_features(X6)

#Scaling_Of_Y_Label

min_y = Y.min()
max_y = Y.max()
scale_y_numerator = Y-min_y
scale_y_denominator = max_y-min_y
scale_y = scale_y_numerator/scale_y_denominator


m,col = 155,6
ones = np.ones((m,1))
after_scaling_of_features = np.concatenate((ones,scale_x1,scale_x2,scale_x3,scale_x4,scale_x5,scale_x6), axis=1)
theta = np.zeros((7,1))
iterations = 200000
alpha = 0.01

# Defining Cost function

def Get_cost_J(X,Y,Theta):
    Pridictions = np.dot(X,Theta)
    Error = Pridictions-Y
    SqrError = np.power(Error,2)
    SumSqrError = np.sum(SqrError)
    J  = (1/2*m)*SumSqrError # Where m is tototal number of rows
    return J

#Defining Gradient Decent Algorithm

def Gradient_Decent_Algo(X,Y,Theta,alpha,itrations,m):
    histroy = np.zeros((itrations,1))
    for i in range(itrations):
        temp =(np.dot(X,Theta))-Y
        temp = (np.dot(X.T,temp))*alpha/m
        Theta = Theta - temp
        histroy[i] = Get_cost_J(X, Y, Theta)
       
    return (histroy,Theta)

h,t = Gradient_Decent_Algo(after_scaling_of_features, scale_y, theta, alpha, iterations, m)

#Values Of Theta

t0 = t[0,0]
t1 = t[1,0]
t2 = t[2,0]
t3 = t[3,0]
t4 = t[4,0]
t5 = t[5,0]
t6 = t[6,0]

#Now For Testing Of 50 Rows

testing_x1 = CSV_Data.iloc[155:206,0:1]
testing_x2 = CSV_Data.iloc[155:206,1:2]
testing_x3 = CSV_Data.iloc[155:206,2:3]
testing_x4 = CSV_Data.iloc[155:206,3:4]
testing_x5 = CSV_Data.iloc[155:206,4:5]
testing_x6 = CSV_Data.iloc[155:206,5:6]
testing_y = CSV_Data.iloc[155:206,6:7]

#Scaling Of Testing Features

scale_testing_x1 = scaling_of_X_features(testing_x1)
scale_testing_x2 = scaling_of_X_features(testing_x2)
scale_testing_x3 = scaling_of_X_features(testing_x3)
scale_testing_x4 = scaling_of_X_features(testing_x4)
scale_testing_x5 = scaling_of_X_features(testing_x5)
scale_testing_x6 = scaling_of_X_features(testing_x6)

#Now we find out prediction of testing data

oness = np.ones((50,1))
concatenation_of_features = np.concatenate((oness,scale_testing_x1,scale_testing_x2,scale_testing_x3,scale_testing_x4,scale_testing_x5,scale_testing_x6), axis=1)
prediction_of_testing_features = np.dot(concatenation_of_features,t)

min_test_y = testing_y.min()
max_test_y = testing_y.max()

final_prediction_salary_price = np.zeros([50,1])
for i in range(0,50):
    final_prediction_salary_price[i,0] = prediction_of_testing_features[i,0]*(max_test_y-min_test_y)+min_test_y
print(final_prediction_salary_price)    





