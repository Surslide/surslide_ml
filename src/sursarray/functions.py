"""Public functional interface for sursarray.

This module provides functions for creating, manipulating, and performing
operations on arrays without requiring an 'Array' instance.

Functions accept scalars, booleans, lists, tuples, or 'Array' instances and provide
a functional alternative to the object-oriented interface.

See Also
--------
surslide_ml.sursarray.array
    Multidimensional array class and its methods.
surslide_ml.sursarray.core
    Internal functions of sursarray responsible for data processing.
"""

from .array import Array
__all__ = ["shape", "sum", "mean", "max", "min", "flatten", "reshape", "abs", "sqrt", "argsort"]
ScalarOrArray = int | float | bool | list | tuple | Array

def shape(array : ScalarOrArray) -> tuple:
    """Determine the dimensions of an array or scalar.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    
    Returns
    -------
    tuple
        Shape of the inserted array.
    
    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.shape
        Attribute of the instance Array that represents the dimensions of the array.

    Examples
    --------
    >>> shape([[2, 3], [4, 5]])
    (2, 2)
    >>> shape([])
    (0,)
    >>> shape(10)
    ()
    """
    if isinstance(array, (Array)):
        return array.shape
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).shape
    else:
        raise TypeError("'shape' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' type in parameter 'array'")
    
def sum(array : ScalarOrArray, axis : None | int = None) -> Array | int | float | bool:
    """Return the sum of scalars in an array over the specified axis.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    axis : None | int, optional
        Determines the dimension that the function will affect.
    
    Returns
    -------
    int | float | bool | Array
        The sum of scalars over the specified axis.
    
    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.sum
        Method of the instance Array that returns the sum of scalars in an array over the specified axis.

    Examples
    --------
    >>> sum([[2, 3], [4, 5]], -1)
    Array([5, 9])
    >>> sum([10, 20, 30, 40, 50, 60, 70, 80, 90], None)
    450
    >>> sum([[1000, 500], [70, 39], [68, 89]], 0)
    Array([1138, 628])
    """
    if isinstance(array, Array):
        return array.sum(axis)
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).sum(axis)
    else:
        raise TypeError("'sum' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' type in parameter 'array'")
    
def mean(array : ScalarOrArray, axis : None | int = None) -> Array | float:
    """Return the mean of scalars in an array over the specified axis.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    axis : None | int, optional
        Determines the dimension that the function will affect.
    
    Returns
    -------
    float | Array
        The mean of scalars over the specified axis.
    
    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.mean
        Method of the instance Array that returns the mean of scalars in an array over the specified axis.

    Examples
    --------
    >>> mean([[2, 3], [4, 5]], -1)
    Array([2.5, 4.5])
    >>> mean([10, 20, 30, 40, 50, 60, 70, 80, 90], None)
    50.0
    >>> mean([[1000, 500], [70, 39], [68, 89]], 0)
    Array([379.33333333, 209.33333333])
    """
    if isinstance(array, Array):
        return array.mean(axis)
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).mean(axis)
    else:
        raise TypeError("'mean' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' type in parameter 'array'")

def max(array : ScalarOrArray, axis : None | int = None) -> Array | int | float | bool:
    """Return the max of scalars in an array over the specified axis.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    axis : None | int, optional
        Determines the dimension that the function will affect.
    
    Returns
    -------
    int | float | bool | Array
        The max of scalars over the specified axis.
    
    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.max
        Method of the instance Array that returns the max of scalars in an array over the specified axis.

    Examples
    --------
    >>> max([[2, 3], [4, 5]], -1)
    Array([3, 5])
    >>> max([10, 20, 30, 40, 50, 60, 70, 80, 90], None)
    90
    >>> max([[1000, 500], [70, 39], [68, 89]], 0)
    Array([1000, 500])
    """
    if isinstance(array, Array):
        return array.max(axis)
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).max(axis)
    else:
        raise TypeError("'max' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' type in parameter 'array'")

def min(array : ScalarOrArray, axis : None | int = None) -> Array | int | float | bool:
    """Return the min of scalars in an array over the specified axis.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    axis : None | int, optional
        Determines the dimension that the function will affect.
    
    Returns
    -------
    int | float | bool | Array
        The min of scalars over the specified axis.

    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.min
        Method of the instance Array that returns the min of scalars in an array over the specified axis.

    Examples
    --------
    >>> min([[2, 3], [4, 5]], -1)
    Array([2, 4])
    >>> min([10, 20, 30, 40, 50, 60, 70, 80, 90], None)
    10
    >>> min([[1000, 500], [70, 39], [68, 89]], 0)
    Array([68, 39])
    """
    if isinstance(array, Array):
        return array.min(axis)
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).min(axis)
    else:
        raise TypeError("'min' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' type in parameter 'array'")
    
def flatten(array : ScalarOrArray) -> list:
    """Return each scalar in the array, in a one dimension list. 

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    
    Returns
    -------
    list
        A one-dimensional list containing all scalar elements.

    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.flatten
        Method of the instance Array that returns each scalar in the array, in a one dimension list.

    Examples
    --------
    >>> flatten([[2, -7], [-4, 5]])
    [2, -7, -4, 5]
    >>> flatten([0, 10, -20, -30, 40, -50, 60, 70, 80, 90])
    [0, 10, -20, -30, 40, -50, 60, 70, 80, 90]
    >>> flatten([[1000, -500], [70, -39], [-68, -89]])
    [1000, -500, 70, -39, -68, -89]
    """
    if isinstance(array, (Array)):
        return array.flatten()
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).flatten()
    else:
        raise TypeError("'flatten' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' type in parameter 'array'")

def reshape(array : ScalarOrArray, new_shape : tuple | list | int) -> Array:
    """Change the shape of the array into the selected shape.

    Maintain the same scalars as the original array.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    new_shape : tuple | list | int
        The shape of the new array that will be returned.

    Returns
    -------
    Array
        The inserted array changed to match the new shape.

    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.reshape
        Method of the instance Array that changes the shape of the array into the selected shape.

    Examples
    --------
    >>> reshape([[2, 3], [4, 5]], -1)
    Array([2, 3, 4, 5])
    >>> reshape([10, 20, 30, 40, 50, 60, 70, 80, 90], (3, 3))
    Array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    >>> reshape([[1000, 500], [70, 39], [68, 89]], [2, -50])
    Array([[1000, 500, 70], [39, 68, 89]])
    """
    if isinstance(array, Array):
        return array.reshape(new_shape)
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).reshape(new_shape)
    else:
        raise TypeError("'reshape' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' type in parameter 'array'")

def abs(array : ScalarOrArray) -> Array | int | float:
    """Return the absolute value of scalars in an array.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    
    Returns
    -------
    int | float | Array
        The absolute value of scalars in the array.

    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.abs
        Method of the instance Array that returns the absolute value of scalars in an array.

    Examples
    --------
    >>> abs([[2, -7], [-4, 5]])
    Array([[2, 7], [4, 5]])
    >>> abs([0, 10, -20, -30, 40, -50, 60, 70, 80, 90])
    Array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> abs([[1000, -500], [70, -39], [-68, -89]])
    Array([[1000, 500], [70, 39], [68, 89]])
    """
    if isinstance(array, Array):
        return array.abs()
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).abs()
    else:
        raise TypeError("'abs' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' types")

def sqrt(array : ScalarOrArray) -> Array | int | float:
    """Return the square root of scalars in an array.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    
    Returns
    -------
    int | float | Array
        The square root of scalars in the array.
    
    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.__pow__
        Dunder method of the instance Array that returns exponentiation between Arrays or between an Array and a scalar.

    Examples
    --------
    >>> sqrt([[4, 9], [16, 49]])
    Array([[2, 3], [4, 7]])
    >>> sqrt(400)
    20
    >>> sqrt([[1000, 500], [70, 39], [68, 89]])
    Array([[31.6227766, 22.36067977], [8.36660027, 6.244998], [8.24621125, 9.43398113]])
    """
    if isinstance(array, (Array, int, float, bool)):
        return array ** 0.5
    elif isinstance(array, (list, tuple)):
        return Array(array) ** 0.5
    else:
        raise TypeError("'sqrt' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' types")

def argsort(array : ScalarOrArray, axis : None | int = -1) -> Array:
    """Sort the elements over the specified axis of the array and return their indexes.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    axis : None | int, optional
        Determines the dimension that the function will affect.
        
    Returns
    -------
    Array
        An array with the indexes of the sorted elements.
    
    Raises
    ------
    TypeError
        If the inserted array is not a scalar, bool, Array, list, or tuple type.

    See Also
    --------
    Array.argsort
        Method of the instance Array that sorts the elements over the specified axis of the array and return their indexes.

    Examples
    --------
    >>> argsort([[2, 3], [4, 5]], -1)
    Array([[0, 1], [0, 1]])
    >>> argsort([10, 20, 30, 40, 50, 60, 70, 80, 90], None)
    Array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> argsort([[1000, 500], [70, 39], [68, 89]], 0)
    Array([[2, 1], [1, 2], [0, 0]])
    """
    if isinstance(array, Array):
        return array.argsort(axis)
    elif isinstance(array, (list, tuple, int, float, bool)):
        return Array(array).argsort(axis)
    else:
        raise TypeError("'argsort' expected 'int', 'float', 'bool', 'list, 'tuple', or 'Array' types")
