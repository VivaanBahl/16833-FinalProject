import numpy as np

def no_force(th,x,y):
	return [0,0,0]

def radial_waves(th,x,y):
    fth = (np.pi/36) #swirl counter-clockwise
    fx = np.sin(x)*np.cos(np.power(x,2) + np.power(y,2))
    fy = np.sin(y)*np.cos(np.power(x,2) + np.power(y,2))
    return np.array([fth,fx,fy])

def linear(th,x,y):
    fth = th
    fx = x/20.0
    fy = y/20.0
    return np.asarray([fth,fx,fy])