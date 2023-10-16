import sys
from utilities import index_to_label

def filter(bloch,n):
    for mu in range(4**n):
         label = index_to_label(mu,n)
         ny = len([x for x in label if x==2])
         if(ny%2==1):
            print(mu,label,ny)
            bloch[mu,:] = (0,0)
    return bloch

