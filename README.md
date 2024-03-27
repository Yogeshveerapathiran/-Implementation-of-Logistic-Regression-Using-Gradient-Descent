# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.
   
3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.
 
5.Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value. 

 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: yogesh
RegisterNumber:  212222040185
*/
```

## Output:
 import numpy as np
 
import matplotlib.pyplot as plt

from scipy import optimize


data=np.loadtxt("ex2data1.txt",delimiter=',')

X=data[:,[0,1]]

y=data[:,2]


X[:5]


y[:5]


plt.figure()

plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")

plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")

plt.xlabel("Exam 1 score")

plt.ylabel("Exam 2 score")

plt.legend()

plt.show()


def sigmoid(z):

    return 1/(1+np.exp(-z))
    

plt.plot()

X_plot=np.linspace(-10,10,100)

plt.plot(X_plot,sigmoid(X_plot))

plt.show()


def costFunction (theta,X,y):

    h=sigmoid(np.dot(X,theta))
    
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    
    grad=np.dot(X.T,h-y)/X.shape[0]
    
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))

theta=np.array([0,0,0])

J,grad=costFunction(theta,X_train,y)

print(J)

print(grad)


X_train=np.hstack((np.ones((X.shape[0],1)),X))

theta=np.array([-24,0.2,0.2])

J,grad=costFunction(theta,X_train,y)

print(J)

print(grad)


def cost (theta,X,y):

    h=sigmoid(np.dot(X,theta))
    
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    
    return J
    

def gradient (theta,X,y):

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
    
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")

    plt.contour(xx,yy,y_plot,levels=[0])
    
    plt.xlabel("Exam 1 score")
    
    plt.ylabel("Exam 2 score")
    
    plt.legend()
    
    plt.show()
    


plotDecisionBoundary(res.x,X,y)


prob=sigmoid(np.dot(np.array([1,45,85]),res.x))

print(prob)


def predict(theta,X):

    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    
    prob=sigmoid(np.dot(X_train,theta))
    
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X)==y)

## OUTPUT:

               Array Value of x
![10](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/7231c9ad-17d3-4659-bbad-7740d6f90abc)

               Array Value of y
![9](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/f4b2050d-8891-4f66-91e9-f249ee4d5cc0)

                   Exam 1 - score graph
![8](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/7818182a-05b3-4a33-9ec2-6dbfe2d6b609)

                 Sigmoid function graph
![7](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/b5bc2990-8b00-4fa2-9592-28a0405f41a6)

               X_train_grad value
![6](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/6136dfc2-3706-48da-80db-feed1ea47946)

             Y_train_grad value
![5](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/74f26772-6fe0-423f-8a43-ada42225b7d4)

                    Print res.x
![4](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/d2f33050-7c4e-4224-96fa-c9bdb04a400f)

             Decision boundary - graph for exam score
![3](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/8ee1ad97-4a1e-4b67-b18f-d7f69b99ceb1)

               Proability value
![2](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/37344dc5-bc0f-4210-84ca-9e74d8fac581)

               Prediction value of mean
![1](https://github.com/RITHISHlearn/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446645/469742fc-2b97-498c-9993-ab13c37cb606)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

