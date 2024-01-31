import numpy as np

base = int(input("base: "))
arr = np.zeros(base+1)

tot = pow(base,base)

def checkEq(base, arr):
  if arr[0]==1:
    arr2 = arr[1:]
    for a in arr2:
      if a!=0:
        return False
    return True

tot_p=0
n_unique = np.zeros(base)

while not checkEq(base,arr):
  carry=True
  i = len(arr)-1
  while carry==True and i>=0:
    carry=False
    arr[i]+=1
    if arr[i] == base:
      arr[i] = 0
      carry = True
      i-=1
  #print(arr)
  n_unique[np.unique(arr[1:]).shape[0]-1] += 1

print(tot)
print(n_unique)

Ns = 0
for ti,t in enumerate(n_unique):
  Ns+=t/tot*(ti+1)
print(Ns/(base*base))
print(np.sum(n_unique))
