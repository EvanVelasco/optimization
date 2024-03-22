import numpy as np
import cvxpy as cp


def select_cols(matrix: np.ndarray, cols: list):
    '''
    Selects a subset of columns from a given matrix.

    Arguments:
        matrix (np.ndarray) - Matrix to choose from
        cols (list) - list of indicies that correspond to which column vectors to keep
    
    Returns: Numpy array of chosen columns.
    '''
    i,j,k = cols[0],cols[1],cols[2]
    return np.array([[row[i],row[j],row[k]] for row in matrix])


def cutting_stock(A: np.ndarray, b: np.ndarray, w: np.ndarray, W: float):
    '''
    Solves the cutting stock problem with a Column Generation algorithm using Numpy and CVXPY

    Arguments:
        A (np.ndarray) - Matrix utilized in the Restricted Master Problem
        b (np.ndarray) - Vector of constraints for RMP
        w (np.ndarray) - width of desired cuts
        W (float) - Total width
    
    Returns: tuple of (1,2,3)
        1. Optimal value the objective value
        2. Value of x which minimizes the objective
        3. Optimal basis matrix
    '''
    min_reduced_cost = -1000
    c = [1 for _ in b]

    while min_reduced_cost < 0:
        # Set Up RMP base case
        x = cp.Variable(A.shape[1])

        constraints = [A @ x == b,
                    x >= 0]

        obj = cp.Minimize(cp.sum(x))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS)

        cols_for_basis = [i for i,value in enumerate(np.isclose(x.value, 0, atol=1e-03)) if value==False]
        basis = select_cols(A, cols_for_basis)
        inv_basis = np.linalg.inv(basis)
        dual_sol = np.dot(c, inv_basis)


        # Set Up Pricing Problem
        a = cp.Variable(len(b), integer=True)
        constraints = [w @ a <= W,
                a >= 0]
        obj = cp.Maximize(dual_sol @ a)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        min_reduced_cost = 1 - prob.value
        A = np.hstack((A, np.atleast_2d(a.value).T))

    return (prob.value, x.value, basis)




# Example Use Case
A = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])

b = np.array([150,200,300])
w = np.array([25,35,45])
W = 100

result = cutting_stock(A,b,w,W)

print('----------------------------------------------------')
print(f'Optimal objective value: {round(result[0],5)}')
print(f'Optimal solution of x is: {[round(i,5) for i in result[1] if round(i,5) != 0]}')
print(f'Optimal basis of x is: \n{result[2]}')
print('----------------------------------------------------')