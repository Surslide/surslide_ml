"""Internal functions of sursarray responsible for data processing.

This module provides core functions that implement array operations,
including indexing, sorting, and calculations.

Notes
-----
This is an internal module, end users should not interact with this module directly.
Functions that starts with '_' should be used only in this module.

See Also
--------
surslide_ml.sursarray.array
    Multidimensional array class and its methods.
surslide_ml.sursarray.functions
    Public functional interface for sursarray.
"""

__all__ = ["internal_shape", "internal_flatten", "internal_reshape", "extremity", "internal_mean", "internal_sum", 
           "internal_abs", "internal_argsort", "set_or_get_index", "array_calculation"]
ScalarOrArray = list | tuple | int | float | bool

#Dimension related functions ---------------
def internal_shape(array : ScalarOrArray) -> tuple:
    """Determine the dimensions of an array or scalar.

    Validate that the array is homogeneous and contains only scalars, booleans, lists, or tuples.
    
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
    ValueError
        If the inserted array is a ragged array.
    TypeError
        If the inserted array or its elements is not scalar, bool, list, or tuple types.
    
    Examples
    --------
    >>> internal_shape([[2, 3], [4, 5]])
    (2, 2)
    >>> internal_shape([])
    (0,)
    >>> internal_shape(10)
    ()
    """
    if not isinstance(array, (list, tuple)):
        if isinstance(array, (int, float, bool)):
            return ()
        raise TypeError("Shape only supports 'int', 'float', 'list', 'tuple', or 'bool' types")
    array_shape = [len(array)]
    if array_shape == [0]:
        return (0,)
    element = array[0]

    #Just looking first element shape.
    while isinstance(element, (list, tuple)):
        element_size = len(element)
        array_shape.append(element_size)
        if element_size == 0:
            break
        element = element[0]

    def validate_shape(array, shape):
    #Checks if the estimated shape is correct.
        n_dim = len(shape)
        if isinstance(array, (int, float, bool)):
            if n_dim != 0:
                raise ValueError("Different array elements size")
            return
        elif isinstance(array, (list, tuple)):
            if n_dim == 0:
                raise ValueError("Different array elements size")
            current_shape = shape[0]
            remaining_shape = shape[1:]
            size_array = len(array)
            if current_shape != size_array:
                raise ValueError("Different array elements size")
            if n_dim == 1:
                if size_array == 0:
                    if current_shape != 0:
                        raise ValueError("Different array elements size")
                    return
            for element in array:
                validate_shape(element, remaining_shape)
            return
        raise TypeError("Shape only supports 'int', 'float', 'list', 'tuple', or 'bool' element types")
        
    validate_shape(array, array_shape)
    return tuple(array_shape)

def internal_flatten(array : ScalarOrArray) -> list: 
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
        If the inserted array or its elements is not scalar, bool, list, or tuple types.

    See Also
    --------
    _size
        Return the total number of scalars in the array.
    _recursive_copy
        Return a deep copy of the inserted array.

    Examples
    --------
    >>> internal_flatten([[2, 3], [4, 5]])
    [2, 3, 4, 5]
    >>> internal_flatten([])
    []
    >>> internal_flatten(10)
    [10]
    """
    if not isinstance(array, (int, float, bool, list, tuple)):
        raise TypeError("Flatten only supports 'int', 'float', 'list', 'tuple', or 'bool' types")
    copied_array = _recursive_copy(array) #used to avoid getting two vectors in the same memory place
    if _size(array) == 0:
        return []
    if isinstance(array, (int, float, bool)):
        return [array]
    def recursive_flatten(array):
        if isinstance(array[0], (int, float, bool)): #_size and __init__ already verifies shape mismatch
            return array
        elif isinstance(array[0], (list, tuple)):
            result = []
            for item in array:
                result.extend(recursive_flatten(item))
            return result
        raise TypeError("Flatten only supports 'int', 'float', 'list', 'tuple', or 'bool' element types")
    return recursive_flatten(copied_array)

def internal_reshape(array : ScalarOrArray, new_shape : tuple | list | int) -> list | float | int | bool:
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
    list | float | int | bool
        The inserted array changed to match the new shape.
    
    Raises
    ------
    ValueError
        If the new shape is incompatible or has multiple negative values.
    TypeError
        If the inserted new shape is not int, list, or tuple.

    See Also
    --------
    _size
        Return the total number of scalars in the array.
    internal_flatten
        Return each scalar in the array, in a one dimension list.
    
    Examples
    --------
    >>> internal_reshape([[2, 3], [4, 5]], -1)
    [2, 3, 4, 5]
    >>> internal_reshape([10, 20, 30, 40, 50, 60, 70, 80, 90], (3, 3))
    [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    >>> internal_reshape([[1000, 500], [70, 39], [68, 89]], [2, -50])
    [[1000, 500, 70], [39, 68, 89]]
    """
    if not isinstance(new_shape, list):
        if isinstance(new_shape, int):
            new_shape = [new_shape]
        elif isinstance(new_shape, tuple):
            new_shape = list(new_shape)
        else:
            raise TypeError("Reshape accepts only 'int' or a 'list'/'tuple' that contains 'int' as a new shape")
    
    previous_size = _size(array)
    new_size = 1
    adapt_count = 0
    for idx, element in enumerate(new_shape):
        if element < 0:
            adapt_idx = idx
            adapt_count += 1
            continue
        new_size *= element

    if 0 in {new_size, previous_size}:
        raise ValueError(f"{new_shape} is not a valid shape for the array")
    if adapt_count > 1:
        raise ValueError("Only one automatic dimension (negative number) allowed in reshape")
    elif adapt_count == 1:
        adapt_val = previous_size / new_size
        if adapt_val % 1 != 0:
            raise ValueError(f"{new_shape} is not a valid shape for the array")
        new_shape[adapt_idx] = int(adapt_val)
    elif new_size != previous_size:
        raise ValueError(f"{new_shape} is not a valid shape for the array")
    
    array_flat = internal_flatten(array)

    def recursive_reshape(new_shape):
        if new_shape:
            return [recursive_reshape(new_shape[1:]) for _ in range(new_shape[0])]
        return array_flat.pop(0)

    return recursive_reshape(new_shape)  
#Functions for methods ----------------
def extremity(array : ScalarOrArray, extreme_type : str, array_ndim : int, axis : int | None) -> int | float | bool | list:
    """Return the max or min of scalars in an Array over the specified axis.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    extreme_type : str
        Determines if it will be returned the max or min value of the array.
    array_ndim : int
        Number of dimensions the array has.
    axis : int | None
        Determines the dimension that the function will affect.
    
    Returns
    -------
    int | float | bool | list
        The max or min over the specified axis.
    
    Raises
    ------
    ValueError
        If array is empty or the axis is out of bounds for inserted array.
    TypeError
        if axis is not an int or None.    
    
    See Also
    --------
    _size
        Return the total number of scalars in the array.
    
    Examples
    --------
    >>> extremity([[2, 3], [4, 5]], "max", 2, -1)
    [3, 5]
    >>> extremity([10, 20, 30, 40, 50, 60, 70, 80, 90], "max", 1, None)
    90
    >>> extremity([[1000, 500], [70, 39], [68, 89]], "min", 2, 0)
    [68, 39]
    """
    if (not isinstance(axis, int)) and (axis is not None):
        raise TypeError(f"Axis must be an int or None")
    if isinstance(array, (int, float, bool)) and axis in {-1,0,None}:
        return array
    if axis is not None:
        if axis < 0:
            axis += array_ndim
        if not (0 <= axis < array_ndim):
            raise ValueError(f"Inserted axis is out of bounds for array of dimension '{array_ndim}'")
    if _size(array) == 0:
        raise ValueError("An empty array doesn't have 'max' or 'min'")

    def recursive_extremity(array, extreme_type, axis):
        result = float("-inf") if extreme_type == "max" else float("inf")

        if isinstance(array[0], (int, float, bool)):
            for element in array:
                result = get_extremity(result, element, extreme_type)
            return result
        elif isinstance(array[0], (list, tuple)):
            if axis is None:
                for element in array:
                    attempt = recursive_extremity(element, extreme_type, axis)
                    result = get_extremity(result, attempt, extreme_type)
                return result
            elif axis == 0:
                def recursive_extremity_axis0(current_max, competitor_max, extreme_type):
                    if isinstance(current_max, (int, float, bool)):
                        return get_extremity(current_max, competitor_max, extreme_type)
                    return [recursive_extremity_axis0(a, b, extreme_type) for a, b in zip(current_max, competitor_max)]
                result = array[0]
                for element in array[1:]:
                    result = recursive_extremity_axis0(result, element, extreme_type)
                return result
            return [recursive_extremity(element, extreme_type, axis - 1) for element in array]

    def get_extremity(val1, val2, extreme_type):
        if extreme_type == "max":
            return val1 if val1 > val2 else val2
        return val1 if val1 < val2 else val2

    return recursive_extremity(array, extreme_type, axis)

def internal_mean(array : ScalarOrArray, array_ndim : int, axis : int | None) -> float | list:
    """Return the mean of scalars in an Array over the specified axis.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    array_ndim : int
        Number of dimensions the array has.
    axis : int | None
        Determines the dimension that the function will affect.
    
    Returns
    -------
    float | list
        The mean of scalars over the specified axis.

    See Also
    --------
    internal_sum
        Return the sum of scalars in an Array over the specified axis.
    _size
        Return the total number of scalars in the array.
    internal_shape
        Determine the dimensions of an array or scalar.
    array_calculation
        Calculate new arrays.

    Examples
    --------
    >>> internal_mean([[2, 3], [4, 5]], 2, -1)
    [2.5, 4.5]
    >>> internal_mean([10, 20, 30, 40, 50, 60, 70, 80, 90], 1, None)
    50.0
    >>> internal_mean([[1000, 500], [70, 39], [68, 89]], 2, 0)
    [379.3333333333333, 209.33333333333334]
    """
    result = internal_sum(array, array_ndim, axis)
    divisor = _size(array) if axis is None else internal_shape(array)[axis]
    return array_calculation(result, divisor, "div", False, internal_shape(result), internal_shape(divisor))

def internal_sum(array : ScalarOrArray, array_ndim : int, axis : int | None) -> int | float | bool | list:
    """Return the sum of scalars in an Array over the specified axis.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    array_ndim : int
        Number of dimensions the array has.
    axis : int | None
        Determines the dimension that the function will affect.
    
    Returns
    -------
    int | float | bool | list
        The sum of scalars over the specified axis.
    
    Raises
    ------
    ValueError
        If axis is out of bounds for inserted array.
    TypeError
        if axis is not an int or None.    

    See Also
    --------
    _size
        Return the total number of scalars in the array.
    internal_shape
        Determine the dimensions of an array or scalar.
    array_calculation
        Calculate new arrays.

    Examples
    --------
    >>> internal_sum([[2, 3], [4, 5]], 2, -1)
    [5, 9]
    >>> internal_sum([10, 20, 30, 40, 50, 60, 70, 80, 90], 1, None)
    450
    >>> internal_sum([[1000, 500], [70, 39], [68, 89]], 2, 0)
    [1138, 628]
    """
    if (not isinstance(axis, int)) and (axis is not None):
        raise TypeError(f"Axis must be an int or None")
    if isinstance(array, (int, float, bool)) and axis in {-1,0,None}:
        return array
    if axis is not None:
        if axis < 0:
            axis += array_ndim
        if not (0 <= axis < array_ndim):
            raise ValueError(f"Inserted axis is out of bounds for array of dimension '{array_ndim}'")
    if _size(array) == 0 and axis not in {None, array_ndim - 1}:
        return []

    def recursive_sum(array, axis):
        if isinstance(array, (int, float, bool)):
            return array
        result = 0
        if axis == 0:
            for element in array:
                result = array_calculation(result, element, "add", False, internal_shape(result), internal_shape(element))
            return result
        elif axis is None:
            for element in array:
                result += recursive_sum(element, axis)
            return result
        return [recursive_sum(element, axis - 1) for element in array]
    return recursive_sum(array, axis)

def internal_abs(array : ScalarOrArray) -> int | float | list:
    """Return the absolute value of all numbers in an Array.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    
    Returns
    -------
    int | float | list
        The absolute value of all numbers in an Array.
    
    Examples
    --------
    >>> internal_abs([[40.9, -30.2], [-500.4, 490]])
    [[40.9, 30.2], [500.4, 490]]
    >>> internal_abs([99, -3])
    [99, 3]
    >>> internal_abs(-200)
    200
    """
    if isinstance(array, (int, float, bool)):
        return abs(array)
    return [internal_abs(element) for element in array]

def internal_argsort(array : ScalarOrArray, axis : int | None, array_shape : tuple, array_ndim : int) -> list:
    """Sort the elements over the specified axis of the array and return their indexes.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    axis : int | None
        Determines the dimension that the function will affect.
    array_shape : tuple
        The shape of array.
    array_ndim : int
        Number of dimensions the array has.

    Returns
    -------
    list
        An array with the indexes of the sorted elements.

    Raises
    ------
    ValueError
        If axis is out of bounds for inserted array.
    TypeError
        if axis is not an int or None. 

    See Also
    --------
    internal_flatten
        Return each scalar in the array, in a one dimension list. 
    _recursive_copy
        Return a deep copy of the inserted array.
    set_or_get_index
        Sets an index as a value or gets the value of an index.

    Examples
    --------
    >>> internal_argsort([[2, 3], [4, 5]], 1, (2, 2), 2)
    [[0, 1], [0, 1]]
    >>> internal_argsort([10, 20, 30, 40, 50, 60, 70, 80, 90], None, (9,), 1)
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> internal_argsort([[1000, 500], [70, 39], [68, 89]], 0, (3, 2), 2)
    [[2, 1], [1, 2], [0, 0]]
    """
    if (not isinstance(axis, int)) and (axis is not None):
        raise TypeError(f"Axis must be an int or None")
    if isinstance(array, (int, float, bool)) and axis in {-1,0,None}:
        return [0]
    if axis is not None:
        if axis < 0:
            axis += array_ndim
        if not (0 <= axis < array_ndim):
            raise ValueError("Axis out of bounds")
    
    def recursive_argsort(array, axis, element_shape):
        if axis is None or len(element_shape) == 0:
            array_flat = internal_flatten(array)
            return sorted(range(len(array_flat)), key=lambda i: array_flat[i])
        if axis > 0:
            return [recursive_argsort(element, axis - 1, element_shape[1:]) for element in array]
        copied_array = _recursive_copy(array)
        indexes_values = {}
        def arg_generate_dict(array, idx_element=()):
            if isinstance(array, (int, float)):
                if idx_element not in indexes_values:
                    indexes_values[idx_element] = []
                indexes_values[idx_element].append(array)
                return
            for idx, element in enumerate(array):
                arg_generate_dict(element, idx_element + (idx,))
        for element in array:
            arg_generate_dict(element)
        for key_val in indexes_values.items():
            sorted_indexes = recursive_argsort(key_val[1], None, ())
            for idx, element in enumerate(copied_array):
                set_or_get_index(element, key_val[0], element_shape, sorted_indexes[idx])
        return copied_array

    return recursive_argsort(array, axis, array_shape[1:])
#Functions for dunder methods ---------------
def set_or_get_index(array : ScalarOrArray, indexes : tuple | int | slice | list, array_shape : tuple, new_value : None | ScalarOrArray = None) -> None | ScalarOrArray:
    """Set an index as a value or get the value of an index.
    
    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    indexes : tuple | int | slice | list
        Indexes that determines where set or get will affect the array
    array_shape : tuple
        The shape of array.
    new_value : ScalarOrArray | None, optional
        The value to be set
    
    Returns
    -------
    None | int | float | bool | list | tuple
        Set returns None and get can return any array elements
    
    See Also
    --------
    internal_shape
        Determine the dimensions of an array or scalar.
    _recursive_copy
        Return a deep copy of the inserted array.
    
    Examples
    --------
    >>> set_or_get_index([[2, 3], [4, 5]], (1,), (2, 2))
    [4, 5]
    >>> set_or_get_index([10, 20, 30, 40, 50, 60, 70, 80, 90], ([3, 6],), (9,), 200)
    None
    >>> set_or_get_index([[1000, 500], [70, 39], [68, 89]], (slice(0, 2), ), (3, 2))
    [[1000, 500], [70, 39]]
    """ 

    if not isinstance(indexes, tuple):
        indexes = (indexes,)
    array_dim_qty = len(array_shape)
    if len(indexes) > array_dim_qty:
        raise ValueError("Too many indexes")
    if new_value is not None:
        shape_other = internal_shape(new_value)
        element_shape = array_shape[len(indexes):]
        if element_shape != shape_other:
            raise ValueError(f"New value of shape '{shape_other}' can't be set in an element with shape '{element_shape}'")
    while len(indexes) < array_dim_qty:
        indexes += (slice(None),)

    if new_value is not None:
        def recursive_set(array, indexes, new_value):
            current_index = indexes[0]
            remaining_indexes = indexes[1:]
            last_index = (len(indexes) == 1)
            
            if isinstance(current_index, int):
                if last_index:
                    array[current_index] = _recursive_copy(new_value)
                else:
                    recursive_set(array[current_index], remaining_indexes, new_value)
            elif isinstance(current_index, slice):
                slice_args = current_index.indices(len(array))
                if last_index:
                    for idx in range(slice_args[0], slice_args[1], slice_args[2]):
                        array[idx] = _recursive_copy(new_value)
                else:
                    for idx in range(slice_args[0], slice_args[1], slice_args[2]):
                        recursive_set(array[idx], remaining_indexes, new_value)
            elif isinstance(current_index, list):
                if last_index:
                    for idx in current_index:
                        array[idx] = _recursive_copy(new_value)
                else:
                    for idx in current_index:
                        recursive_set(array[idx], remaining_indexes, new_value)
            else:
                raise ValueError("Invalid index")
        recursive_set(array, indexes, new_value)
        return
    def recursive_get(array, indexes):
        if len(indexes) == 0:
            return array
        current_index = indexes[0]
        remaining_indexes = indexes[1:]

        if isinstance(current_index, int):
            return recursive_get(array[current_index], remaining_indexes)
        elif isinstance(current_index, slice):
            return [recursive_get(element, remaining_indexes) for element in array[current_index]]
        elif isinstance(current_index, list):
            return [recursive_get(array[idx], remaining_indexes) for idx in current_index]
        raise ValueError("Invalid index")      
    return recursive_get(array, indexes)

def array_calculation(array : ScalarOrArray, other : ScalarOrArray, operation_type : str, reverse : bool, array_shape : tuple, other_shape : tuple) -> list | int | float:
    """Calculate new arrays.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.
    other : ScalarOrArray
        An array of any rank, or a scalar value.
    operation_type : str
        Determines the operation that will be done.
    reverse : bool
        Determines the order of operation.
    array_shape : tuple
        The shape of array.
    other_shape : tuple
        The shape of other.

    Returns
    -------
    list | int | float
        A new array calculated by operation between array and other.

    Raises
    ------
    ValueError
        If array and other does not have the same shape and none of them are a scalar.
    TypeError
        If array or other have not scalar, bool, list, tuple element types.

    Examples
    --------
    >>> array_calculation([[0.5, 15], [8, 2.5]], [[2, 3], [4, 5]], "div", True, (2, 2), (2, 2))
    [[4.0, 0.2], [0.5, 2.0]]
    >>> array_calculation([9, -3], 2, "pow", False, (2,), ())
    [81, 9]
    >>> array_calculation(-200, [10, 20, 30, 40, 50, 60, 70, 80, 90], "add", False, (), (1,))
    [-190, -180, -170, -160, -150, -140, -130, -120, -110]
    """

    if (not isinstance(array, (list, tuple, int, float, bool))) or (not isinstance(other, (list, tuple, int, float, bool))):
        raise TypeError("Array and other needs to be 'int', 'float', 'list', 'tuple', or 'bool' types")
    if isinstance(array, (list, tuple)) and isinstance(other, (list, tuple)) and array_shape != other_shape:
        raise ValueError("Array and other needs to have the same shape or atleast one of them need to be a scalar")
    
    def recursive_calculation(array, other, operation_type, reverse):
        if isinstance(other, (int, float, bool)):
            if isinstance(array, (int, float, bool)):
                return apply_operation(array, other, operation_type, reverse)
            elif isinstance(array, (list, tuple)):
                return [recursive_calculation(val1, other, operation_type, reverse) for val1 in array]

        elif isinstance(other, (list, tuple)):
            if isinstance(array, (int, float)):
                return [recursive_calculation(array, val2, operation_type, reverse) for val2 in other]
            elif isinstance(array, (list, tuple)):
                return [recursive_calculation(val1, val2, operation_type, reverse) for val1, val2 in zip(array, other)]
        raise TypeError("elements inside of array and other needs to be 'int', 'float', 'list', 'tuple', or 'bool' types")
    
    def apply_operation(array, value, operation_type, reverse):
        if reverse:
            if operation_type == "sub":
                return value - array
            elif operation_type == "div":
                return value / array
            elif operation_type == "pow":
                if value < 0 and array % 1 != 0:
                    raise ValueError("Negative number can't be raised to a decimal power")
                return value ** array   
        if operation_type == "add":
            return array + value
        elif operation_type == "mul":
            return array * value
        elif operation_type == "sub":
            return array - value
        elif operation_type == "div":
            return array / value
        elif operation_type == "pow":
            if array < 0 and value % 1 != 0:
                raise ValueError("Negative number can't be raised to a decimal power")
            return array ** value
        elif operation_type == "eq":
            return array == value
        elif operation_type == "ne":
            return array != value
        
    return recursive_calculation(array, other, operation_type, reverse)
#Auxiliary functions that should not be imported
def _recursive_copy(array : ScalarOrArray) -> list | int | float | bool:
    """Return a deep copy of the inserted array.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.

    Returns
    -------
    list | int | float | bool
        A deep copy of the inserted array.

    Examples
    --------
    >>> _recursive_copy([[2, 3], [4, 5]])
    [[2, 3], [4, 5]]
    >>> _recursive_copy([])
    []
    >>> _recursive_copy(10)
    10
    """
    if isinstance(array, (int, float, bool)):
        return array
    return [_recursive_copy(element) for element in array]

def _size(array : ScalarOrArray) -> int:
    """Return the total number of scalars in the array.

    Parameters
    ----------
    array : ScalarOrArray
        An array of any rank, or a scalar value.

    Returns
    -------
    int
        Number of scalars in the array.

    See Also
    --------
    internal_shape
        Determine the dimensions of an array or scalar.

    Examples
    --------
    >>> _size([[2, 3], [4, 5]])
    4
    >>> _size([])
    0
    >>> _size(10)
    1
    """
    result = 1
    for element in internal_shape(array):
        result *= element
    return result
