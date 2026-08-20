def f(x):
    """Defined function to apply Newton Raphson method to"""
    return x * x + 2 * x + 1


def d(x, f, eps=0.00000001):
    """Derivative funtion"""
    return (f(x + eps) - f(x)) / eps


def dd(x, f, eps=0.00000001):
    """Second Derivative funtion"""
    return (d(x + eps, f) - d(x, f)) / eps


def neuton_raphson(x_0, f, eps=0.00001, thresh=0.00001):
    """Newton Raphson method"""
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")

    if not isinstance(x_0, (int, float)):
        raise TypeError("x0` must be numeric")

    x_t_1 = x_0
    x_t = x_t_1 - d(x_t_1, f, eps) / dd(x_t_1, f, eps)

    while abs(x_t - x_t_1) > eps:
        temp = x_t_1 - d(x_t_1, f, eps) / dd(x_t_1, f, eps)
        x_t = temp - d(temp, f, eps) / dd(temp, f, eps)
        x_t_1 = temp

    return x_t
