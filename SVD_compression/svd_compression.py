import numpy as np
import matplotlib.pyplot as plt


X = np.loadtxt("tech.txt")

def low_rank_approx(A,k):
    U,S,Vh = np.linalg.svd(A)

    U_k = U[:,:k]
    S_k = np.diag(S[:k])
    Vh_k = Vh[:k,:]

    low_rank = np.dot(U_k,np.dot(S_k, Vh_k))

    return low_rank


ranks = [5,15,25,50]
for rank in ranks:
    plt.gray()
    plt.imshow(low_rank_approx(X,k=rank))
    plt.title(f'Rank {rank} Approximation')
    plt.savefig(f'images/rank_{rank}.jpeg')
    plt.show()


