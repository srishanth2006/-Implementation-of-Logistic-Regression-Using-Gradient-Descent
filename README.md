# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b
6. importing the required modules from sklearn.
Obtain the graph. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
### Array value of X:
![Screenshot 2024-04-14 131806](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/053e9ac6-179b-447b-a5a7-f2e9df459fd2)

### Array value of Y:

![Screenshot 2024-04-14 131817](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/2b71e34a-55f9-4fb0-9e9c-3e76ade09e20)
### Exam 1-Score graph:

![Screenshot 2024-04-14 131836](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/fac44219-c277-4bde-bf01-4dbdea0b9d19)


### Sigmoid function graph:
![Screenshot 2024-04-14 131847](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/8589a85e-957b-41b1-ac0b-293d8934d7b8)
### X_Train_grad value:


![Screenshot 2024-04-14 131902](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/ce6b44ad-e2af-4ac3-879c-3dfc64f273bd)

### Y_Train_grad value:
![Screenshot 2024-04-14 131913](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/87535761-3b04-47ba-bcde-dce9187ef8aa)

### Print res.X:
![Screenshot 2024-04-14 131926](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/9f90ad12-bdfd-49f4-b8cc-ed0cae4e244f)

### Decision boundary-gragh for exam score:
![Screenshot 2024-04-14 131926](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/225c0464-aebf-4774-a67a-06407624741a)

### Probability value:
![Screenshot 2024-04-14 131943](https://github.com/srishanth2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150319470/f00da078-8c45-4d62-b113-f0215e7e4d9c)

### Prediction value of mean:


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

