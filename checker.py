import numpy as np
N = 5
M = 3
D = 4

a=np.array([])
b=np.array([])
for i in range(N*M):
    a=np.append(a,i)
print(a)
for i in range(N*D):
    b=np.append(b,2*i)
print(b)
a=a.reshape((M,N))
b=b.reshape((N,D))
print(a.dot(b))
