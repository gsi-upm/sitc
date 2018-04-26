import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from scipy.constants import golden, pi

def gen_spiral_dataset(n_examples=500, n_classes=2, a=None, b=None, pi_space=3):
    n_spirals = n_classes
    
    # default: golden spiral
    if a is None:
        a = golden
    if b is None:
        b = 2/pi

    theta = np.linspace(0,pi_space*pi, num=n_examples)
    xy = np.zeros((n_examples,2))

    # logaritmic spirals
    x_golden_parametric = lambda a, b, theta: a**(theta*b) * cos(theta)
    y_golden_parametric = lambda a, b, theta: a**(theta*b) * sin(theta)
    x_golden_parametric = np.vectorize(x_golden_parametric)
    y_golden_parametric = np.vectorize(y_golden_parametric)

    # rotation matrix
    gen_rotation = lambda theta: np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])

    # rotation angles
    rot_division = (2*pi) / n_spirals
    rot_thetas = [i * rot_division for i in range(n_spirals)]

    XY = np.zeros((2, n_examples, n_spirals))
    for i in range(n_spirals):
        x = x_golden_parametric(a, b, theta)
        y = y_golden_parametric(a, b, theta)
        xy = np.vstack((x,y))
        R = gen_rotation(rot_thetas[i])
        xy_ = np.dot(R.T, xy)
        XY[:,:,i] = xy_
    
    return XY

def load_spiral_dataset(n_examples=300, n_classes=2):
    XY = gen_spiral_dataset(n_examples, n_classes)
    X_s = []
    y_s = []
    for i in range(XY.shape[2]):
        X = XY[:,:,i].T
        X_s.append(X)
        y = np.array([i] * XY.shape[1]).T
        y_s.append(y)
    X = np.vstack(X_s)
    y = np.hstack(y_s)
    
    return X, y

def plot_dataset(X,y):
    cm = plt.cm.RdBu
    plt.scatter(X[:,0], X[:,1], c=y, cmap=cm, lw=.5, s=10)
    
def plot_decision_surface(X, y, classifier, h=0.02):
    cm = plt.cm.RdBu
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    

    z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])#[:, 1]
    
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
    
    plot_dataset(X, y)
