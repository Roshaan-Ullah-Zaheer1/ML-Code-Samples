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

#making it 2 Rank
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
iterations = 200000
alpha = 0.01

#Concatenation

X_scaling = np.concatenate((ones,scale_x1,scale_x2,scale_x3,scale_x4,scale_x5,scale_x6,scale_x7,scale_x8,scale_x9,scale_x10,scale_x11,scale_x12,scale_x13,scale_x14,scale_x15,scale_x16,scale_x17,scale_x18,scale_x19,scale_x20,scale_x21,scale_x22,scale_x23,scale_x24,scale_x25,scale_x26,scale_x27,scale_x28,scale_x29,scale_x30),axis=1)

#Scaling_Of_Y

min_y = Y.min()
max_y = Y.max()
scale_y_numerator = Y-min_y
scale_y_denominator = max_y-min_y
scale_y = scale_y_numerator/scale_y_denominator

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

h,t = Gradient_Decent_Algo(X_scaling, scale_y, theta, alpha, iterations, m)

final_prediction = np.dot(X_scaling,t)
final_prediction = final_prediction*(max_y-min_y)+min_y

plt.scatter(X,Y)
plt.plot(X,final_prediction,color="red")
plt.show()


# =============================================================================
# #Now Predict the salary of year 2020
# 
# #now scaling year 2020
# 
# xnew_numerator = 2020-min_x
# xnew_denomenator = max_x-min_x
# x_new = xnew_numerator/xnew_denomenator
# x_new2 = np.power(x_new,2)
# x_new3 = np.power(x_new,3)
# x_new4 = np.power(x_new,4)
# x_new5 = np.power(x_new,5)
# x_new6 = np.power(x_new,6)
# x_new7 = np.power(x_new,7)
# x_new8 = np.power(x_new,8)
# x_new9 = np.power(x_new,9)
# x_new10 = np.power(x_new,10)
# x_new11 = np.power(x_new,11)
# x_new12 = np.power(x_new,12)
# x_new13 = np.power(x_new,13)
# x_new14 = np.power(x_new,14)
# x_new15 = np.power(x_new,15)
# 
# t0 = t[0,0]
# t1 = t[1,0]
# t2 = t[2,0]
# t3 = t[3,0]
# t4 = t[4,0]
# t5 = t[5,0]
# t6 = t[6,0]
# t7 = t[7,0]
# t8 = t[8,0]
# t9 = t[9,0]
# t10 = t[10,0]
# t11 = t[11,0]
# t12 = t[12,0]
# t13 = t[13,0]
# t14 = t[14,0]
# t15 = t[15,0]
# 
# predict_new_value = t0+(t1*x_new)+(t2*x_new2)+(t3*x_new3)+(t4*x_new4)+(t5*x_new5)+(t6*x_new6)+(t7*x_new7)+(t8*x_new8)+(t9*x_new9)+(t10*x_new10)+(t11*x_new11)+(t12*x_new12)+(t13*x_new13)+(t14*x_new14)+(t15*x_new15)
# 
# salary_predicted_final = predict_new_value*(max_y-min_y)+min_y
# 
# print(salary_predicted_final)
# 
# =============================================================================
