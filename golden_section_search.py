
def golden_search(func, xl, xu, epsilon=0.0001):
    '''
    Golden Section Search algorithm to find local minimum of a univariate function.

    Arguemnts:
        func - function to minimize
        xl - lower bound of search area
        xu - upper bound of search area
        epsilon - algorithm stops when xu - xl is smaller than this number
    '''
    from math import sqrt
    assert xu > xl, 'Upper bound xu must be larger than lower bound xl'

    phi = (1 + sqrt(5)) / 2

    x1 = xu - (xu-xl) / phi
    x2 = xl + (xu-xl) / phi

    fx1, fx2 = func(x1), func(x2)

    while abs(xu-xl) >= epsilon:
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
