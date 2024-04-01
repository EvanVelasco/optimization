import numpy as np
import matplotlib.pyplot as plt


X = np.loadtxt("clownImage.txt")

U,S,Vh = np.linalg.svd(X)


def low_rank_approx(A,k):
    U,S,Vh = np.linalg.svd(A)

    U_k = U[:,:k]
    S_k = np.diag(S[:k])
    Vh_k = Vh[:k,:]

    low_rank = np.dot(U_k,np.dot(S_k, Vh_k))

    return low_rank


ranks = [5,15,25]
for rank in ranks:
    plt.gray()
    plt.imshow(low_rank_approx(X,k=rank))
    plt.title(f'Rank {rank} Approximation')
    plt.savefig(f'rank_{rank}.jpeg')
    plt.show()
