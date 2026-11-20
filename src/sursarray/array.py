"""Multidimensional array class and its methods.

This module provides the 'Array' class, which represents multidimensional
arrays and implements methods and dunder methods for array manipulation,
such as indexing, arithmetic operations, comparisons, and calculations.

See Also
--------
surslide_ml.sursarray.functions
    Public functional interface for sursarray.
surslide_ml.sursarray.core
    Internal functions of sursarray responsible for data processing.
"""
from __future__ import annotations
from collections.abc import Iterator
from .core import (internal_shape, internal_flatten, internal_reshape, extremity, internal_mean, 
                   internal_sum, internal_abs, internal_argsort ,set_or_get_index, array_calculation)
__all__ = ["Array"]
ArrayLike = list | tuple | int | float | bool

class Array:
    """Represent a multidimensional array.

    Provides an object-oriented interface for storing and manipulating
    multidimensional data.

    Parameters
    ----------
    data : ArrayLike
        Data used to initialize the array.

    Attributes
    ----------
    data : ArrayLike
        Data stored in the array.
    shape : tuple
        Dimensions of the array.
    ndim : int
        Number of dimensions.
    size : int
        Total number of scalar elements.
    """
    def __init__(self, data : ArrayLike) -> None:
        """Initialize an Array instance.

        Parameters
        ----------
        data : ArrayLike
            Data used to create the array.

        Notes
        -----
        Internal shape validates that the array is homogeneous and contains only scalars, booleans, lists, or tuples.

        See Also
        --------
        internal_shape
            Core function to determine the dimensions of an array or scalar.
        """
        if isinstance(data, Array):
            data = data.data
        self.shape = internal_shape(data)
        self.data = data
        self.ndim = len(self.shape) 
        self.size = 1
        for dim in self.shape:
            self.size *= dim

    def __repr__(self) -> str:
        """Return the representation of the array.

        Returns
        -------
        str
            The representation of the array, including its class name.
        """
        return f"Array({self.data})"
    
    def __str__(self) -> str:
        """Return the string representation of the array.

        Returns
        -------
        str
            The string representation of the Array.
        """
        return f"{self.data}"
    
    def __len__(self) -> int:
        """Return the length of the first dimension of the array.

        Returns
        -------
        int
            Number of elements in the first dimension.

        Examples
        --------
        >>> len(Array([[1, 2], [3, 4]]))
        2
        """
        return len(self.data)
    
    def __getitem__(self, indexes : tuple | int | slice | list) -> Array | int | float | bool:
        """Return the element in the respective index.

        Parameters
        ----------
        indexes : tuple | int | slice | list
            Indexes used to get the array element.

        Returns
        -------
        Array | int | float | bool
            The selected element.

        See Also
        --------
        set_or_get_index
            Core function to set an index as a value or get the value of an index.

        Examples
        --------
        >>> a = Array([[1, 2], [3, 4]])
        >>> a[0]
        Array([1, 2])
        >>> a[0, 1]
        2
        """
        result = set_or_get_index(self.data, indexes, self.shape)
        return Array(result) if isinstance(result, (list, tuple)) else result
    
    def __setitem__(self, indexes : tuple | int | slice | list, new_value : Array | ArrayLike) -> None:
        """Change the value in the respective index.

        Parameters
        ----------
        indexes : tuple | int | slice | list
            Indexes used to select the position to be changed.
        new_value : Array | ArrayLike
            The value to be set.

        See Also
        --------
        set_or_get_index
            Core function to set an index as a value or get the value of an index.

        Examples
        --------
        >>> a = Array([[1, 2], [3, 4]])
        >>> a[0, 1] = 10
        >>> a
        Array([[1, 10], [3, 4]])
        """
        if isinstance(new_value, Array):
            set_or_get_index(self.data, indexes, self.shape, new_value.data)
            return
        set_or_get_index(self.data, indexes, self.shape, new_value)

    def __iter__(self) -> Iterator[Array | int | float | bool]:
        """Provide an iterator over the elements of the array.

        Yields
        ------
        Array | int | float | bool
            If the array is multi-dimensional, yields elements as 'Array' instances. 
            If the array is one-dimensional, yields scalar elements.
        """
        for item in self.data:
            yield Array(item) if isinstance(item, (list, tuple)) else item
    
    def __add__(self, other : Array | ArrayLike) -> Array:
        """Return addition between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The sum of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.
            
        Examples
        --------
        >>> Array([1, 2, 3]) + 2
        Array([3, 4, 5])
        >>> Array([1, 2]) + Array([3, 4])
        Array([4, 6])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "add", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "add", False, self.shape, internal_shape(other))
        return Array(result)

    def __radd__(self, other : Array | ArrayLike) -> Array:
        """Return reverse addition between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The sum of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) + 2
        Array([3, 4, 5])
        >>> Array([1, 2]) + Array([3, 4])
        Array([4, 6])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "add", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "add", False, self.shape, internal_shape(other))
        return Array(result) 

    def __sub__(self, other : Array | ArrayLike) -> Array:
        """Return subtraction between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The difference of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) - 2
        Array([-1, 0, 1])
        >>> Array([1, 2]) - Array([3, 4])
        Array([-2, -2])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "sub", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "sub", False, self.shape, internal_shape(other))
        return Array(result)

    def __rsub__(self, other : Array | ArrayLike) -> Array:
        """Return reverse subtraction between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The difference of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) - 2
        Array([1, 0, -1])
        >>> Array([1, 2]) - Array([3, 4])
        Array([2, 2])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "sub", True, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "sub", True, self.shape, internal_shape(other))
        return Array(result)

    def __mul__(self, other : Array | ArrayLike) -> Array:
        """Return multiplication between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The product of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) * 2
        Array([2, 4, 6])
        >>> Array([1, 2]) * Array([3, 4])
        Array([3, 8])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "mul", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "mul", False, self.shape, internal_shape(other))
        return Array(result)

    def __rmul__(self, other : Array | ArrayLike) -> Array:
        """Return reverse multiplication between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The product of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) * 2
        Array([2, 4, 6])
        >>> Array([1, 2]) * Array([3, 4])
        Array([3, 8])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "mul", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "mul", False, self.shape, internal_shape(other))
        return Array(result)
    
    def __truediv__(self, other : Array | ArrayLike) -> Array:
        """Return division between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The quotient of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) / 2
        Array([0.5, 1, 1.5])
        >>> Array([1, 2]) / Array([3, 4])
        Array([0.33333333, 0.5])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "div", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "div", False, self.shape, internal_shape(other))
        return Array(result)

    def __rtruediv__(self, other : Array | ArrayLike) -> Array:
        """Return reverse division between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The quotient of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) / 2
        Array([2, 1, 0.66666667])
        >>> Array([1, 2]) / Array([3, 4])
        Array([3, 2])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "div", True, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "div", True, self.shape, internal_shape(other))
        return Array(result)

    def __pow__(self, other : Array | ArrayLike) -> Array:
        """Return exponentiation between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The power of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) ** 2
        Array([1, 4, 9])
        >>> Array([1, 2]) ** Array([3, 4])
        Array([1, 16])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "pow", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "pow", False, self.shape, internal_shape(other))
        return Array(result)

    def __rpow__(self, other : Array | ArrayLike) -> Array:
        """Return reverse exponentiation between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array
            The power of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) ** 2
        Array([2, 4, 8])
        >>> Array([1, 2]) ** Array([3, 4])
        Array([3, 16])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "pow", True, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "pow", True, self.shape, internal_shape(other))
        return Array(result)

    def __neg__(self) -> Array:
        """Return the negative form of the array.

        Returns
        -------
        Array
            The negative form of the array.

        See Also
        --------
        __mul__
            Return multiplication between Arrays or between an Array and a scalar.

        Examples
        --------
        >>> -Array([1, 2, 3])
        Array([-1, -2, -3])
        >>> -Array([1, 2])
        Array([-1, -2])
        """
        return self * -1

    def __eq__(self, other : Array | ArrayLike) -> Array | bool:
        """Return equality comparison between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array | bool
            The bool result of equality comparison of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) == 2
        Array([False, True, False])
        >>> Array([1, 2]) == Array([1, 4])
        Array([True, False])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "eq", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "eq", False, self.shape, internal_shape(other))
        return Array(result) if isinstance(result, list) else result

    def __ne__(self, other : Array | ArrayLike) -> Array | bool:
        """Return non-equality comparison between Arrays or between an Array and a scalar.

        Parameters
        ----------
        other : Array | ArrayLike
            An array of any rank, or a scalar value.

        Returns
        -------
        Array | bool
            The bool result of non-equality comparison of the arrays or scalar and array.

        See Also
        --------
        array_calculation
            Core function to calculate new arrays.
        internal_shape
            Core function to determine the dimensions of an array or scalar.

        Examples
        --------
        >>> Array([1, 2, 3]) != 2
        Array([True, False, True])
        >>> Array([1, 2]) != Array([1, 4])
        Array([False, True])
        """
        if isinstance(other, Array):
            result = array_calculation(self.data, other.data, "ne", False, self.shape, other.shape)
        else:
            result = array_calculation(self.data, other, "ne", False, self.shape, internal_shape(other))
        return Array(result) if isinstance(result, list) else result

    def sum(self, axis : None | int = None) -> Array | int | float | bool:
        """Return the sum of scalars in an array over the specified axis.

        Parameters
        ----------
        axis : None | int, optional
            Determines the dimension that the function will affect.
        
        Returns
        -------
        int | float | bool | Array
            The sum of scalars over the specified axis.
        
        See Also
        --------
        internal_sum
            Core function to return the sum of scalars in an array over the specified axis.

        Examples
        --------
        >>> Array([[2, 3], [4, 5]]).sum(-1)
        Array([5, 9])
        >>> Array([10, 20, 30, 40, 50, 60, 70, 80, 90]).sum(None)
        450
        >>> Array([[1000, 500], [70, 39], [68, 89]]).sum(0)
        Array([1138, 628])
        """
        result = internal_sum(self.data, self.ndim, axis)
        return Array(result) if isinstance(result, list) else result

    def mean(self, axis : None | int = None) -> Array | float:
        """Return the mean of scalars in an array over the specified axis.
    
        Parameters
        ----------
        axis : None | int, optional
            Determines the dimension that the function will affect.
        
        Returns
        -------
        float | Array
            The mean of scalars over the specified axis.
        
        See Also
        --------
        internal_mean
            Core function to return the mean of scalars in an array over the specified axis.
    
        Examples
        --------
        >>> Array([[2, 3], [4, 5]]).mean(-1)
        Array([2.5, 4.5])
        >>> Array([10, 20, 30, 40, 50, 60, 70, 80, 90]).mean(None)
        50.0
        >>> Array([[1000, 500], [70, 39], [68, 89]]).mean(0)
        Array([379.33333333, 209.33333333])
        """
        result = internal_mean(self.data, self.ndim, axis)
        return Array(result) if isinstance(result, list) else result
    
    def max(self, axis : None | int = None) -> int | float | bool | Array:
        """Return the max of scalars in an array over the specified axis.
    
        Parameters
        ----------
        axis : None | int, optional
            Determines the dimension that the function will affect.
        
        Returns
        -------
        int | float | bool | Array
            The max of scalars over the specified axis.
        
        See Also
        --------
        extremity
            Core function to return the max or min of scalars in an Array over the specified axis.
    
        Examples
        --------
        >>> Array([[2, 3], [4, 5]]).max(-1)
        Array([3, 5])
        >>> Array([10, 20, 30, 40, 50, 60, 70, 80, 90]).max(None)
        90
        >>> Array([[1000, 500], [70, 39], [68, 89]]).max(0)
        Array([1000, 500])
        """
        result = extremity(self.data, "max", self.ndim, axis)
        return Array(result) if isinstance(result, list) else result
    
    def min(self, axis : None | int = None) -> int | float | bool | Array:
        """Return the min of scalars in an array over the specified axis.
    
        Parameters
        ----------
        axis : None | int, optional
            Determines the dimension that the function will affect.
        
        Returns
        -------
        int | float | bool | Array
            The min of scalars over the specified axis.
        
        See Also
        --------
        extremity
            Core function to return the max or min of scalars in an Array over the specified axis.
    
        Examples
        --------
        >>> Array([[2, 3], [4, 5]]).min(-1)
        Array([2, 4])
        >>> Array([10, 20, 30, 40, 50, 60, 70, 80, 90]).min(None)
        10
        >>> Array([[1000, 500], [70, 39], [68, 89]]).min(0)
        Array([68, 39])
        """
        result = extremity(self.data, "min", self.ndim, axis)
        return Array(result) if isinstance(result, list) else result

    def flatten(self) -> list:
        """Return each scalar in the array, in a one dimension list.
        
        Returns
        -------
        list
            A one-dimensional list containing all scalar elements.
        
        See Also
        --------
        internal_flatten
            Core function to return each scalar in the array, in a one dimension list.
    
        Examples
        --------
        >>> Array([[2, 3], [4, 5]]).flatten()
        [2, 3, 4, 5]
        >>> Array([10, 20, 30, 40, 50, 60, 70, 80, 90]).flatten()
        [10, 20, 30, 40, 50, 60, 70, 80, 90]
        >>> Array([[1000, 500], [70, 39], [68, 89]]).flatten()
        [1000, 500, 70, 39, 68, 89]
        """
        return internal_flatten(self.data)

    def reshape(self, new_shape : tuple | list | int) -> Array:
        """Change the shape of the array into the selected shape.

        Maintain the same scalars as the original array.

        Parameters
        ----------
        new_shape : tuple | list | int
            The shape of the new array that will be returned.

        Returns
        -------
        Array
            The inserted array changed to match the new shape.

        See Also
        --------
        internal_reshape
            Core function to change the shape of the array into the selected shape.

        Examples
        --------
        >>> Array([[2, 3], [4, 5]], ).reshape(-1)
        [2, 3, 4, 5]
        >>> Array([10, 20, 30, 40, 50, 60, 70, 80, 90]).reshape((3, 3))
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        >>> Array([[1000, 500], [70, 39], [68, 89]]).reshape([2, -50])
        [[1000, 500, 70], [39, 68, 89]]
        """
        return Array(internal_reshape(self.data, new_shape))

    def abs(self) -> int | float | Array:
        """Return the absolute value of scalars in an array.

        Returns
        -------
        int | float | Array
            The absolute value of scalars in the array.

        See Also
        --------
        internal_abs
            Core function to return the absolute value of scalars in an array.

        Examples
        --------
        >>> Array([[2, -7], [-4, 5]]).abs()
        Array([[2, 7], [4, 5]])
        >>> Array([0, 10, -20, -30, 40, -50, 60, 70, 80, 90]).abs()
        Array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        >>> Array([[1000, -500], [70, -39], [-68, -89]]).abs()
        Array([[1000, 500], [70, 39], [68, 89]])
        """
        result = internal_abs(self.data)
        return Array(result) if isinstance(result, list) else result

    def argsort(self, axis : None | int = -1) -> Array:
        """Sort the elements over the specified axis of the array and return their indexes.

        Parameters
        ----------
        axis : None | int, optional
            Determines the dimension that the function will affect.
            
        Returns
        -------
        Array
            An array with the indexes of the sorted elements.
        
        See Also
        --------
        internal_argsort
            Core function to sort the elements over the specified axis of the array and return their indexes.

        Examples
        --------
        >>> Array([[2, 3], [4, 5]]).argsort(-1)
        Array([[0, 1], [0, 1]])
        >>> Array([10, 20, 30, 40, 50, 60, 70, 80, 90]).argsort(None)
        Array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        >>> Array([[1000, 500], [70, 39], [68, 89]]).argsort(0)
        Array([[2, 1], [1, 2], [0, 0]])
        """
        return Array(internal_argsort(self.data, axis, self.shape, self.ndim))
