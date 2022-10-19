import numpy as np

if __name__ == '__main__':
    data = np.loadtxt('houses.txt',delimiter=',',dtype=int)
    a = data[:,1]
    b = data[:,2]
    m = np.zeros((data.shape[0],data.shape[1]))
    m[:,0] = data[:,0]
    m[:,1] = b
    m[:,2] = a
    np.savetxt('new_houses.txt',m,delimiter=',')