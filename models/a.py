import numpy as np
import os
import hashlib

def array_fingerprint(arr):
    return hash((arr.shape, hashlib.sha1(arr).hexdigest()))

unique_arrays = set()
results='../Results/'
x=0
for _ in os.listdir(results):
    if _.endswith('.npy'):
        x+=1
        file=os.path.join(results,_)
        data=np.load(file)
        unique_arrays.add(array_fingerprint(data))
        # print(data.shape)

print(x)
print(len(unique_arrays))