import pytest
import numpy as np
import math

import newton_funct as newton

## Important: structure of tests assumes a dictionary with an 'x'
## key as the output. 
def squared(x):
    return x*x

def minus_squared(x):
    return -x*x
    
def test_basic_function():
    assert np.isclose(newton.neuton_raphson(2.95, np.cos), math.pi)

def test_basic_function_1():
    assert np.isclose(newton.neuton_raphson(1, squared), 0)

def test_basic_function_2():
    assert np.isclose(newton.neuton_raphson(100, squared), 0)

def test_bad_input():
    with pytest.raises(TypeError):   
        newton.neuton_raphson(np.cos, np.cos)
    ## Ideally, our function would raise the exception with a useful message.
    with pytest.raises(TypeError, match='`x0` must be numeric'):
        newton.neuton_raphson(np.cos, np.cos)
        
def test_bad_input():
    with pytest.raises(TypeError):   
        newton.neuton_raphson(2, 2)
    ## Ideally, our function would raise the exception with a useful message.
    with pytest.raises(TypeError, match='Argument is not a function, it is of type '):
        newton.neuton_raphson(2, 2)
        
## How to check that a warning is (correctly) emitted:
# def test_warning():
#     with pytest.warns(UserWarning, match='function is not strictly convex'):
#         newton.dd(
       
