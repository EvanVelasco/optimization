from math import sqrt

def golden_section(func: callable, xl: float, xu: float, epsilon: float = 0.0001):
    '''
    Golden Section Search algorithm to find local minimum of a univariate function.
    Returns a float which is the minimizing value of x.

    Arguments:
        func (function) - function to minimize
        xl (float) - lower bound of search area
        xu (float) - upper bound of search area
        epsilon (float) - tolerance level. algorithm stops when xu - xl < epsilon
    '''
    assert xu > xl, 'Upper bound xu must be larger than lower bound xl'

    phi = (1 + sqrt(5)) / 2

    x1 = xu - (xu-xl) / phi
    x2 = xl + (xu-xl) / phi

    fx1, fx2 = func(x1), func(x2)

    while xu-xl >= epsilon:
        if fx1 < fx2:
            xu = x2
            x2 = x1
            x1 = xu-(xu-xl) / phi
            fx2 = fx1
            fx1 = func(x1)
        else:
            xl = x1
            x1 = x2
            x2 = xl + (xu-xl) / phi
            fx1 = fx2
            fx2 = func(x2)

    return 0.5*(xl+xu)
