import numpy as np

if __name__ == '__main__':
    data = np.random.randint(1,100,size = (50,10))
    l = np.random.randint(1,5,size = (50,1))
    data = np.hstack((data,l))

    np.savetxt('donnees.txt',data,delimiter = ',')