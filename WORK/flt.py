import numpy as np 
s = np.float32(0)
for i in range(int(8e7)):
    if i % 10000000==0:
        print(i)
    s+= np.float32(i)
print(s)