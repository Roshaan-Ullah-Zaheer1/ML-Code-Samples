import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot

CSV_Data = pd.read_csv("D:/Semester 6/Machine Learning/Assignment 1/hiring.csv")
CSV_Data = CSV_Data.replace(np.NaN,0)

#print(CSV_Data.shape)

#Separating Features

X = CSV_Data.iloc[:,0:3]
Y = CSV_Data.iloc[:,3]
Y = np.array(Y)
Y = Y[:,np.newaxis]

m,col = X.shape
ones = np.ones((m,1))
X = np.hstack((ones,X))

theta = np.zeros((4,1))

#dot_product_testing = np.dot(X,theta)

iterations = 9000
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

h,t = Gradient_Decent_Algo(X, Y, theta, alpha, iterations, m)

t0 = t[0,0]
t1 = t[1,0]
t2 = t[2,0]
t3 = t[3,0]

#predicting salary of 2 year experience 9 test score and 6 interview score

predict_salary1 = t0+(t1*2)+(t2*9)+(t3*6)
print("Salary Predicted 1($) = ",predict_salary1)

#predicting salary of 12 year experience 10 test score and 10 interview score

predict_salary2 = t0+(t1*12)+(t2*10)+(t3*10)
print("Salary Predicted 2($) = ",predict_salary2)