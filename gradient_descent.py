from sympy import *

def gradient_descent(func: callable, x0: float, alpha: float = 0.01, epsilon: float = 0.0001):
    '''
    Performs Gradient Descent for a univariate function using a fixed step size alpha.
    This function relies on computing the derivative of func using sympy.

    Arguments:
        func (function) - function to minimize
        x0 (float) - initial solution
        alpha (float) - step size. Algorithm may not converge if alpha too large
        epsilon (float) - tolerance level. algorithm stops when ||gradient(xk)|| < epsilon
    
    Returns: float which is the value of x which minimizes the function.
    '''
    x_symb = Symbol('x')
    yprime = func(x_symb).diff(x_symb)
    gradient = lambdify(x_symb, yprime, 'numpy')

    xk = x0
    
    while abs(gradient(xk)) > epsilon:
        d = -gradient(xk)
        xk = xk + alpha*d

    return xk
