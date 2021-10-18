import sys
sys.path.insert(1, 'Project2')

import numpy as np
import perceptron as p
import gradient_descent as gd


def load_data(fname):
  f = open(fname, 'r')
  ctr = 0
  y_str = ''
  for line in f:
    line = line.strip().split(';')
    if ctr == 0:
      x_str = line
    else:
      y_str = line
    ctr+=1
  f.close()
  X = []
  Y = []
  for item in x_str:
    temp = [float(x) for x in item.split(',')]
    X.append(temp)
  if len(y_str)>0:
    for item in y_str:
      temp = int(item)
      Y.append(temp)
  X = np.array(X)
  Y = np.array(Y)
  return X, Y


X,Y = load_data("data_1.txt")
w,b = p.perceptron_train(X,Y)
test_acc = p.perceptron_test(X,Y,w,b)
print("Perceptron:",test_acc)

X,Y = load_data("data_2.txt")
w,b = p.perceptron_train(X,Y)

X,Y = load_data("data_1.txt")
test_acc = p.perceptron_test(X,Y,w,b)
print("Perceptron:",test_acc)


def df_test1(x):
  return np.array([2*x[0]])

x = gd.gradient_descent(df_test1,np.array([5.0]),0.1)   
print("Gradient Descent:", x)


def df_test2(x):
  return np.array([2*x[0],3*x[1]*x[1]])

x = gd.gradient_descent(df_test2,np.array([3.0,1.0]),0.1)   
print("Gradient Descent:", x)
