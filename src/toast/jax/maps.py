import jax
from jax import vmap, lax
from jax import numpy as jnp
from copy import deepcopy
from types import EllipsisType
from inspect import Signature, Parameter

# TODO: common error that should be caught with a nice error message
# - passing the wrong number of arguments to the functions (missing some)
#
# set f"{interval_axis}_length" to the iterval_max_length axis name for easier use?
# (used int he template offset operator)
#
# can I catch:
# - errors with the order of inputs? -> we can at least catch inputs with incorect shape / type

#----------------------------------------------------------------------------------------
# PYTREE FUNCTIONS

def is_pytree_leaf(structure):
    """
    Determines if the given structure is a leaf of a pytree.
    
    Args:
        structure: The structure to check.
    
    Returns:
        True if the structure is a leaf, False otherwise.
    """
    if isinstance(structure, jax.Array):
        # An array is considered a leaf
        return True
    if isinstance(structure, list):
        # A list with no sublists, dictionaries, or tuples is a leaf
        return not any(isinstance(elem, (list, dict, tuple)) for elem in structure)
    # All other types are not leaves
    return False

def map_pytree_leaves(func, structure, func_single_values=lambda v: v):
    """
    Applies a function to all leaves in the pytree.
    
    Args:
        func: The function to apply to the leaves.
        structure: The pytree to apply the function to.
        func_single_values: The function to apply to single values (non-pytree elements).
    
    Returns:
        The transformed pytree.
    """
    if is_pytree_leaf(structure):
        return func(structure)
    elif isinstance(structure, dict):
        return {k: map_pytree_leaves(func, v, func_single_values) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [map_pytree_leaves(func, v, func_single_values) for v in structure]
    elif isinstance(structure, tuple):
        return tuple(map_pytree_leaves(func, v, func_single_values) for v in structure)
    else:
        return func_single_values(structure)

def map2_pytree_leaves(func, pytree1, pytree2, func_single_values=lambda v1, v2: v1):
    """
    Applies a function to corresponding leaves of two pytrees.

    Args:
        func: The function to apply to the leaves.
        pytree1: The first pytree.
        pytree2: The second pytree.
        func_single_values: The function to apply to single values (non-pytree elements).

    Returns:
        The transformed pytree.
    """
    if is_pytree_leaf(pytree1):
        return func(pytree1, pytree2)
    elif isinstance(pytree1, dict):
        return {k: map2_pytree_leaves(func, v1, v2, func_single_values) 
                for ((k, v1), v2) in zip(pytree1.items(), pytree2.values())}
    elif isinstance(pytree1, list):
        return [map2_pytree_leaves(func, v1, v2, func_single_values) 
                for v1, v2 in zip(pytree1, pytree2)]
    elif isinstance(pytree1, tuple):
        return tuple(map2_pytree_leaves(func, v1, v2, func_single_values) 
                     for v1, v2 in zip(pytree1, pytree2))
    else:
        return func_single_values(pytree1, pytree2)

def pytree_to_string(pytree):
    """
    Converts a pytree to a string representation.

    Args:
        pytree: The pytree to convert.

    Returns:
        A string representation of the pytree.
    """
    if is_pytree_leaf(pytree) and isinstance(pytree, list):
        items = ", ".join(pytree_to_string(item) for item in pytree)
        return f"Array[{items}]"
    elif isinstance(pytree, dict):
        items = ", ".join(f"{k}: {pytree_to_string(v)}" for k, v in pytree.items())
        return f"{{{items}}}"
    elif isinstance(pytree, list):
        items = ", ".join(pytree_to_string(item) for item in pytree)
        return f"[{items}]"
    elif isinstance(pytree, tuple):
        items = ", ".join(pytree_to_string(item) for item in pytree)
        return f"({items})"
    elif isinstance(pytree, str):
        return pytree  # Return string without quotes
    elif isinstance(pytree, EllipsisType):
        return "..."
    elif isinstance(pytree, type):
        return pytree.__name__
    else:
        return str(pytree)  # Fallback for other types

#------------------------------------------------

def check_pytree_axis(data, axis, info=""):
    """
    Checks that the data's shape is concordant with the given axis.

    Args:
        data: The data to check.
        axis: The axis to check against.
        info: Additional info to include in error messages.

    Raises:
        AssertionError: If the data's shape does not match the given axis.
    """
    if is_pytree_leaf(axis):
        assert len(axis) == data.ndim, f"{info} axis ({axis}) != {data.shape}"
    elif isinstance(axis, dict):
        assert len(axis) == len(data), f"{info} axis ({axis}) != len(data) ({len(data)})"
        data_items = data.values() if isinstance(data, dict) else data
        for d, (k, a) in zip(data_items, axis.items()):
            check_pytree_axis(d, a, f"{info} {k}:")
    elif isinstance(axis, (list, tuple)):
        assert len(axis) == len(data), f"{info} axis ({axis}) != len(data) ({len(data)})"
        for d, a in zip(data, axis):
            check_pytree_axis(d, a, info)
    # we do not cover the case of single values as they are ssumed to be matching

def find_in_pytree(condition, structure):
    """
    Returns the first element in a pytree for which a condition is True.

    Args:
        condition: A function that evaluates to True or False for a given value.
        structure: The pytree to search through.

    Returns:
        The first element that meets the condition, or None if no such element is found.
    """
    if is_pytree_leaf(structure):
        # Check if the leaf contains a matching element
        for value in structure:
            if condition(value):
                return value
    elif isinstance(structure, (list, tuple, dict)):
        # Check if the container has a matching sub-container
        data = structure.values() if isinstance(structure, dict) else structure
        for element in data:
            result = find_in_pytree(condition, element)
            if result is not None:
                return result
    # No match found
    return None

def map_pytree(func, structure):
    """
    Applies a function to all elements in the leaves of the pytree.

    Args:
        func: The function to apply.
        structure: The pytree to apply the function to.

    Returns:
        The pytree with the function applied to each element in its leaves.
    """
    def map_leaf(leaf):
        return [func(v) for v in leaf]

    return map_pytree_leaves(map_leaf, structure, func)

def filter_pytree(condition, structure):
    """
    Filters elements in the leaves of a pytree based on a condition.

    Args:
        condition: A function that returns True for elements to keep.
        structure: The pytree to filter.

    Returns:
        A new pytree with only the elements that satisfy the condition.
    """
    def filter_leaf(leaf):
        return [v for v in leaf if condition(v)]

    return map_pytree_leaves(filter_leaf, structure)

def index_in_pytree(value, structure):
    """
    Finds the index of a value in all leaves of a pytree.

    Args:
        value: The value to search for.
        structure: The pytree to search within.

    Returns:
        The pytree structure with indices of the value in its leaves, or None where the value is not present.
    """
    def index_leaf(leaf):
        return leaf.index(value) if value in leaf else None

    # Applies the function to all leaves, mapping single non-leaf values to None
    return map_pytree_leaves(index_leaf, structure, lambda x: None)

def replace_in_pytree(value, new_value, structure):
    """
    Replaces all instances of a specified value in the leaves of a pytree with a new value.

    Args:
        value: The value to be replaced.
        new_value: The new value to replace with. Can be a single value or a tuple of values.
        structure: The pytree in which to perform the replacement.

    Returns:
        The pytree with the value replaced.
    """
    new_values = list(new_value) if isinstance(new_value, tuple) else [new_value]

    def replace_leaf(leaf):
        new_leaf = []
        for v in leaf:
            if v == value:
                new_leaf.extend(new_values)
            else:
                new_leaf.append(v)
        return new_leaf

    return map_pytree_leaves(replace_leaf, structure)

#----------------------------------------------------------------------------------------
# INDEXING

# Full slice, equivalent to `:` in `[:]`.
ALL = slice(None, None, None)

def is_valid_key(key):
    """
    Determines if a key is valid for indexing in a pytree.

    Args:
        key: The key to validate.

    Returns:
        True if the key is valid, False otherwise.
    """
    return isinstance(key, (tuple, int, jax.Array))

def get_from_pytree(data, key):
    """
    Retrieves an element from a pytree using a key, equivalent to `data[key]`.

    Args:
        data: The pytree from which to retrieve the element.
        key: The key for indexing.

    Returns:
        The element from the pytree corresponding to the given key.
    """
    def get_leaf(data_leaf, key_leaf):
        return data_leaf[key_leaf] if is_valid_key(key_leaf) else data_leaf

    return map2_pytree_leaves(get_leaf, data, key)

def set_in_pytree(data, key, value):
    """
    Sets an element in a pytree using a key, equivalent to `data[key] = value`.

    Args:
        data: The pytree in which to set the value.
        key: The key for indexing.
        value: The value to set.

    Returns:
        The pytree with the specified value set at the specified key.
    """
    value_keys = map2_pytree_leaves(lambda v, k: (v, k), value, key)

    def set_leaf(data_leaf, vk_leaf):
        value_leaf, key_leaf = vk_leaf
        return data_leaf.at[key_leaf].set(value_leaf) if is_valid_key(key_leaf) else data_leaf

    return map2_pytree_leaves(set_leaf, data, value_keys)

def get_index_from_pytree(data, data_axes, index, index_axis):
    """
    Generates a key by setting the index_axis to index, and retrieves the corresponding element from the pytree.

    Args:
        data: The pytree to index into.
        data_axes: The axes specification of the pytree.
        index: The index to retrieve.
        index_axis: The axis along which to index.

    Returns:
        The element from the pytree at the specified index and axis.
    """
    def list_to_key(axes):
        return tuple(index if a == index_axis else ALL for a in axes)

    key = map_pytree_leaves(list_to_key, data_axes)
    return get_from_pytree(data, key)

def set_index_in_pytree(data, data_axes, index, index_axis, value):
    """
    Generates a key by setting the index_axis to index and sets the corresponding element in the pytree.

    Args:
        data: The pytree in which to set the value.
        data_axes: The axes specification of the pytree.
        index: The index to set.
        index_axis: The axis along which to set the value.
        value: The value to set.

    Returns:
        The pytree with the specified value set at the specified index and axis.
    """
    def list_to_key(axes):
        return tuple(index if a == index_axis else ALL for a in axes)

    key = map_pytree_leaves(list_to_key, data_axes)
    return set_in_pytree(data, key, value)

#----------------------------------------------------------------------------------------
# FUNCTION WRAPING

def args_to_kwargs(args, keys):
    """
    Converts a list of arguments into a dictionary of keyword arguments.

    Args:
        args: A list of arguments.
        keys: A list of keys corresponding to the arguments.

    Returns:
        A dictionary where keys are from 'keys' and values are from 'args'.
    """
    if len(args) != len(keys):
        raise ValueError("The number of arguments and keys must be the same")

    return dict(zip(keys, args))

def kwargs_to_args(kwargs):
    """
    Converts a dictionary of keyword arguments into a list of values.

    Args:
        kwargs: The dictionary of keyword arguments.

    Returns:
        A list of values from the dictionary.
    """
    return list(kwargs.values())

def runtime_check_axis(func, in_axes, out_axes):
    """
    Wraps a function to check its axes against a specification at runtime.

    Args:
        func: The function to wrap.
        in_axes: The input axes specification.
        out_axes: The output axes specification.

    Returns:
        The wrapped function.
    """
    def wrapped_func(*args):
        check_pytree_axis(args, in_axes, info="INPUT:")
        output = func(*args)
        check_pytree_axis(output, out_axes, info="OUTPUT:")
        return output

    return wrapped_func

def set_documentation(func, in_axes, out_axes, reference_func=None):
    """
    Adds documentation to a function following the axis specification.

    Args:
        func: The function to document.
        in_axes: The input axes specification.
        out_axes: The output axes specification.
        reference_func: The reference function for documentation. If None, 'func' is used.

    Returns:
        The function with added documentation.
    """
    reference_func = reference_func or func
    doc = f"Original documentation of {reference_func.__name__}:\n{reference_func.__doc__}\n"
    doc += '\nArgs:\n'
    for key, val in in_axes.items():
        doc += f"    {key}: {pytree_to_string(val)}\n"
    doc += '\nReturns:\n' + '    ' +  pytree_to_string(out_axes)
    func.__doc__ = doc

    parameters = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in in_axes.keys()]
    func.__signature__ = Signature(parameters)
    return func

#----------------------------------------------------------------------------------------
# XMAP

def recursive_xmap(f, in_axes, out_axes):
    """
    Implements 'xmap' by applying 'vmap' recursively.

    Args:
        f: The function to be mapped.
        in_axes: The input axes mappings for 'f'.
        out_axes: The output axes mappings for 'f'.

    Returns:
        The transformed function.
    """
    # Finds the first named output axis
    named_output_axis = find_in_pytree(lambda a: isinstance(a, str), out_axes)

    # If no more named output axes, return the function with runtime axis check
    if named_output_axis is None:
        named_input_axis = find_in_pytree(lambda a: isinstance(a, str), in_axes)
        assert named_input_axis is None, f"Unused input axis: {named_input_axis}. Check output axes."
        return runtime_check_axis(f, in_axes, out_axes)

    # Remove axis from inputs and outputs
    filtered_in_axes = filter_pytree(lambda a: a != named_output_axis, in_axes)
    filtered_out_axes = filter_pytree(lambda a: a != named_output_axis, out_axes)

    # Recursively map over the remaining axes
    f_batched = recursive_xmap(f, filtered_in_axes, filtered_out_axes)

    # Get indices of the axis in inputs and outputs
    in_axes_indices = index_in_pytree(named_output_axis, list(in_axes.values()))
    out_axes_indices = index_in_pytree(named_output_axis, out_axes)

    # Apply vmap to remove the current axis
    f_result = vmap(f_batched, in_axes=in_axes_indices, out_axes=out_axes_indices)
    return runtime_check_axis(f_result, in_axes, out_axes)

def xmap(f, in_axes, out_axes):
    """
    A wrapper function for applying 'xmap' to a given function.

    Args:
        f: The function to apply 'xmap' to.
        in_axes: The input axes mappings.
        out_axes: The output axes mappings.

    Returns:
        callable: The batched and documented version of 'f'.
    """
    # TODO run assertions to insure that inputs are correct

    # Batch the function
    f_batched = recursive_xmap(f, in_axes, out_axes)

    # Add documentation
    return set_documentation(f_batched, in_axes, out_axes, reference_func=f)

#----------------------------------------------------------------------------------------
# IMAP

def imap(f, in_axes, interval_axis, 
         interval_starts, interval_ends, interval_max_length,
         output_name, output_as_input=False):
    """
    Extends xmap to handle intervals with padding and reshaping.

    Args:
        f (callable): The function to be mapped over the intervals.
        in_axes (dict): The axes mappings for the input of `f`.
        interval_axis (str): Axis name used for identifying the interval.
        interval_starts (str): Input name containing the starts of each interval.
        interval_ends (str): Input name containing the ends (exclusive) of each interval.
        interval_max_length (str): Input name containing the maximum length of intervals (static if jitted).
        output_name (str): Input name containing the output value.
        output_as_input (bool): If True, the output value is also used as an input to `f`.

    Returns:
        callable: A transformed function that applies `f` over specified intervals in the input data.
    """
    # TODO run assertions to insure that inputs are correct

    # Define the output axes based on the specified output name in the input axes.
    out_axes = in_axes[output_name]

    # Create unique axis names for internal processing.
    interval_length_axis = f"{interval_axis}_length"
    num_intervals_axis = in_axes[interval_starts][0]

    # Filter the input and output axes to include only relevant dimensions.
    in_axes_inner = filter_pytree(lambda a: isinstance(a, EllipsisType) or a == interval_axis, in_axes)
    in_axes_inner[interval_max_length] = []
    out_axes_inner_interval = filter_pytree(lambda a: isinstance(a, EllipsisType) or a == interval_axis, out_axes)
    out_axes_inner = filter_pytree(lambda a: isinstance(a, EllipsisType), out_axes)

    def inner_function(*args):
        """
        Inner function that operates on each interval, applying `f` or retrieving existing output data.

        This function computes the output for each interval index, either by applying `f`
        or by using the existing output, depending on whether the index is within the interval bounds.
        """
        kwargs = args_to_kwargs(args, in_axes_inner.keys())
        start, end = kwargs[interval_starts], kwargs[interval_ends]
        absolute_index, output_data = kwargs[interval_max_length], kwargs[output_name]
        index = start + absolute_index

        def compute_within_interval():
            # Computes the result of function `f` for indices within the interval.
            input_at_index = get_index_from_pytree(kwargs, in_axes_inner, index, interval_axis)
            for key in [interval_starts, interval_ends, interval_max_length]:
                input_at_index.pop(key, None)
            if not output_as_input:
                input_at_index.pop(output_name, None)
            return f(*kwargs_to_args(input_at_index))

        def compute_outside_interval():
            # Retrieves existing output data for indices outside the interval.
            return get_index_from_pytree(output_data, out_axes_inner_interval, index, interval_axis)

        # NOTE: <= because toast intervals are inclusive
        output_at_index = lax.cond(index <= end, compute_within_interval, compute_outside_interval)
        return output_at_index
    inner_function = runtime_check_axis(inner_function, in_axes_inner, out_axes_inner)

    # Prepare axes for batch processing.
    in_axes_batched = replace_in_pytree(interval_axis, (...), in_axes)
    in_axes_batched[interval_max_length] = [interval_length_axis]
    out_axes_batched = replace_in_pytree(interval_axis, (num_intervals_axis, interval_length_axis), out_axes)
    batched_function = xmap(inner_function, in_axes_batched, out_axes_batched)

    # Define outer function axes based on the original input and output axes.
    in_axes_outer = deepcopy(in_axes)
    out_axes_outer = deepcopy(out_axes)

    def outer_function(*args):
        """
        Outer function that orchestrates the overall interval processing.

        This function sets up the interval indexing and invokes the batched inner function
        to process the entire data structure.
        """
        kwargs = args_to_kwargs(args, in_axes_outer.keys())
        max_length, starts, output = kwargs[interval_max_length], kwargs[interval_starts], kwargs[output_name]
        kwargs[interval_max_length] = jnp.arange(max_length)

        output_interval = batched_function(*kwargs_to_args(kwargs))
        indices_interval = starts[:, None] + jnp.arange(max_length)
        output = set_index_in_pytree(output, out_axes_outer, indices_interval, interval_axis, output_interval)
        return output
    outer_function = runtime_check_axis(outer_function, in_axes_outer, out_axes_outer)

    return set_documentation(outer_function, in_axes, out_axes, reference_func=f)