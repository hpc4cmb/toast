import itertools
from copy import deepcopy
from inspect import Parameter, Signature
from types import EllipsisType

import jax
from jax import lax
from jax import numpy as jnp
from jax import vmap

# ----------------------------------------------------------------------------------------
# PYTREE FUNCTIONS


def make_iterable(data):
    """
    Produce a datastructure that can be iterated
    making sure we are not going to iterate on dictionary keys
    """
    return data.values() if isinstance(data, dict) else data


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
        return {
            k: map_pytree_leaves(func, v, func_single_values)
            for k, v in structure.items()
        }
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
        return {
            k: map2_pytree_leaves(func, v1, v2, func_single_values)
            for ((k, v1), v2) in zip(pytree1.items(), make_iterable(pytree2))
        }
    elif isinstance(pytree1, list):
        return [
            map2_pytree_leaves(func, v1, v2, func_single_values)
            for v1, v2 in zip(pytree1, make_iterable(pytree2))
        ]
    elif isinstance(pytree1, tuple):
        return tuple(
            map2_pytree_leaves(func, v1, v2, func_single_values)
            for v1, v2 in zip(pytree1, make_iterable(pytree2))
        )
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


# ------------------------------------------------


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
    # goes through the axis / data
    if is_pytree_leaf(axis):
        assert (
            len(axis) == data.ndim
        ), f"{info} shape ({data.shape}) does not match provided axis ({pytree_to_string(axis)})"
    elif isinstance(axis, dict):
        assert len(axis) == len(
            data
        ), f"{info} has {len(data)} elements which does not match axis ({pytree_to_string(axis)})"
        for d, (k, a) in zip(make_iterable(data), axis.items()):
            check_pytree_axis(d, a, f"{info} '{k}'")
    elif isinstance(axis, (list, tuple)):
        assert len(axis) == len(
            data
        ), f"{info} has {len(data)} elements which does not match axis ({pytree_to_string(axis)})"
        for i, (d, a) in enumerate(zip(make_iterable(data), axis)):
            check_pytree_axis(d, a, f"{info}[{i}]")
    elif isinstance(axis, type):
        is_single_number_tracer = isinstance(data, jnp.ndarray) and (data.size == 1)
        data_type = (
            data.dtype if is_single_number_tracer else type(data)
        )  # deals with JAX tracers being sorts of arrays
        if jnp.issubdtype(axis, jnp.integer):
            # integer types all batched together to simplify axis writing
            assert jnp.issubdtype(
                data_type, jnp.integer
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
        elif jnp.issubdtype(axis, jnp.floating):
            # float types all batched together to simplify axis writing
            assert jnp.issubdtype(
                data_type, jnp.floating
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
        elif jnp.issubdtype(axis, bool):
            # bool types all batched together to simplify axis writing
            assert jnp.issubdtype(
                data_type, bool
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
        else:
            # other, more general, types
            assert isinstance(
                data, axis
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
    # we do not cover the case of other single values as they are assumed to be matching


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


def where_pytree(condition_tree, true_tree, false_tree):
    """
    Apply a conditional selection to elements in PyTrees (Python trees).

    This function iterates through the elements of PyTrees (structures of nested lists, tuples, or dictionaries)
    and applies the `jnp.where` function based on a condition tree. If the condition is true, elements from the
    true_tree are selected, otherwise elements from the false_tree are chosen.

    NOTE: we suppose that `true_tree` and `false_tree` will have the same shape.
          `condition_tree` is allowed to be simpler (could be a single bool, etc).

    Args:
        condition_tree: A PyTree where leaf nodes are boolean conditions.
        true_tree: A PyTree with the same structure as condition_tree, containing values for when the condition is true.
        false_tree: A PyTree with the same structure as condition_tree, containing values for when the condition is false.

    Returns:
        A PyTree with the same structure as the input PyTrees, containing elements from true_tree or false_tree based on
        the conditions in condition_tree.
    """
    if (
        is_pytree_leaf(condition_tree)
        and is_pytree_leaf(true_tree)
        and is_pytree_leaf(false_tree)
    ):
        return jnp.where(condition_tree, true_tree, false_tree)
    else:
        # Ensure condition_tree is iterable
        if (not isinstance(condition_tree, (list, dict, tuple))) and isinstance(
            true_tree, (list, dict, tuple)
        ):
            condition_tree = itertools.repeat(condition_tree)

        # Apply conditional mapping based on the type of true_tree
        if isinstance(true_tree, dict):
            condition_tree = (
                condition_tree.values()
                if isinstance(condition_tree, dict)
                else condition_tree
            )
            return {
                key: where_pytree(cond, true_val, false_val)
                for ((key, true_val), (false_val, cond)) in zip(
                    true_tree.items(), zip(false_tree.values(), condition_tree)
                )
            }
        elif isinstance(true_tree, list):
            return [
                where_pytree(cond, true_val, false_val)
                for (true_val, (false_val, cond)) in zip(
                    true_tree, zip(false_tree, condition_tree)
                )
            ]
        elif isinstance(true_tree, tuple):
            return tuple(
                where_pytree(cond, true_val, false_val)
                for (true_val, (false_val, cond)) in zip(
                    true_tree, zip(false_tree, condition_tree)
                )
            )
        else:
            return jnp.where(condition_tree, true_tree, false_tree)


# ----------------------------------------------------------------------------------------
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
        return (
            data_leaf.at[key_leaf].set(value_leaf)
            if is_valid_key(key_leaf)
            else data_leaf
        )

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


# ----------------------------------------------------------------------------------------
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
    Converts a dictionary of keyword arguments into a tuple of values.

    Args:
        kwargs: The dictionary of keyword arguments.

    Returns:
        A tuple of values from the dictionary.
    """
    return tuple(kwargs.values())


def runtime_check_axis(func, in_axes, out_axes):
    """
    Wraps a function to check its axes against a specification at runtime.
    Once jitted, this function becomes a no-op, it however helps with debugging.

    Args:
        func: The function to wrap.
        in_axes: The input axes specification.
        out_axes: The output axes specification.

    Returns:
        The wrapped function.
    """

    def wrapped_func(*args):
        check_pytree_axis(args, in_axes, info="Inputs")
        output = func(*args)
        check_pytree_axis(output, out_axes, info="Output")
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
    doc += "\nArgs:\n"
    for key, val in in_axes.items():
        doc += f"    {key}: {pytree_to_string(val)}\n"
    doc += "\nReturns:\n" + "    " + pytree_to_string(out_axes)
    func.__doc__ = doc

    parameters = [
        Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in in_axes.keys()
    ]
    func.__signature__ = Signature(parameters)
    return func


# ----------------------------------------------------------------------------------------
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
        assert (
            named_input_axis is None
        ), f"Unused input axis: {named_input_axis}. Check output axes."
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
    # Batch the function
    f_batched = recursive_xmap(f, in_axes, out_axes)

    # Add documentation
    return set_documentation(f_batched, in_axes, out_axes, reference_func=f)


# ----------------------------------------------------------------------------------------
# IMAP


def imap(
    f,
    in_axes,
    interval_axis,
    interval_starts,
    interval_ends,
    interval_max_length,
    output_name,
    output_as_input=False,
    mask_dummy_work=True,
):
    """
    Extends xmap to handle intervals with padding and reshaping.

    Args:
        f (callable): The function to be mapped over the intervals.
        in_axes (dict): The axes mappings for the input of `f`.
        interval_axis (str): Axis name used for identifying the interval.
        interval_starts (str): Input name containing the starts of each interval.
        interval_ends (str): Input name containing the ends (exclusive) of each interval.
        interval_max_length (str): Input name containing the maximum length of intervals (static if jitted). This is also the name of the corresponding axis.
        output_name (str): Input name containing the output value.
        output_as_input (bool): If True, the output value is also used as an input to `f`.
        mask_dummy_work (bool): If True (default value), will do dummy work out of interval then mask it, otherwise will use a test to skip it.

    Returns:
        callable: A transformed function that applies `f` over specified intervals in the input data.
    """
    # Define the output axes based on the specified output name in the input axes.
    out_axes = in_axes[output_name]

    # Create unique axis names for internal processing.
    interval_length_axis = (
        interval_max_length  # same name as the input that contains it
    )
    num_intervals_axis = in_axes[interval_starts][0]

    # Define inner function axes
    # Filter the input and output axes to include only relevant dimensions.
    in_axes_inner = filter_pytree(
        lambda a: isinstance(a, EllipsisType) or a == interval_axis, in_axes
    )
    in_axes_inner[interval_max_length] = []
    out_axes_inner_interval = filter_pytree(
        lambda a: isinstance(a, EllipsisType) or a == interval_axis, out_axes
    )
    out_axes_inner = filter_pytree(lambda a: isinstance(a, EllipsisType), out_axes)
    # version to be used for the within_interval sub-function
    in_axes_inner_within = deepcopy(in_axes_inner)
    for key in [interval_starts, interval_ends, interval_max_length]:
        in_axes_inner_within.pop(key, None)
    if not output_as_input:
        in_axes_inner_within.pop(output_name, None)

    def inner_function(*args):
        """
        Inner function that operates on each interval, applying `f` or retrieving existing output data.

        This function computes the output for each interval index, either by applying `f`
        or by using the existing output, depending on whether the index is within the interval bounds.
        """
        # gets interval specific inputs
        kwargs = args_to_kwargs(args, in_axes_inner.keys())
        start, end = kwargs[interval_starts], kwargs[interval_ends]
        absolute_index, output_data = kwargs[interval_max_length], kwargs[output_name]
        index = start + absolute_index
        # pop those inputs as they will not be useful anymore
        for key in [interval_starts, interval_ends, interval_max_length]:
            kwargs.pop(key, None)
        if not output_as_input:
            kwargs.pop(output_name, None)

        def compute_within_interval():
            # Computes the result of function `f` for indices within the interval.
            kwargs_at_index = get_index_from_pytree(
                kwargs, in_axes_inner_within, index, interval_axis
            )
            return f(*kwargs_to_args(kwargs_at_index))

        def compute_outside_interval():
            # Retrieves existing output data for indices outside the interval.
            return get_index_from_pytree(
                output_data, out_axes_inner_interval, index, interval_axis
            )

        # NOTE: <= because toast intervals are inclusive
        if mask_dummy_work:
            # compute both sides then use a mask
            output_at_index = where_pytree(
                index <= end, compute_within_interval(), compute_outside_interval()
            )
        else:
            # compute only the side we need at the cost of a test
            output_at_index = lax.cond(
                index <= end, compute_within_interval, compute_outside_interval
            )
        return output_at_index

    inner_function = runtime_check_axis(inner_function, in_axes_inner, out_axes_inner)

    # Prepare axes for batch processing.
    in_axes_batched = replace_in_pytree(interval_axis, (...), in_axes)
    in_axes_batched[interval_max_length] = [interval_length_axis]
    out_axes_batched = replace_in_pytree(
        interval_axis, (num_intervals_axis, interval_length_axis), out_axes
    )
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
        # gets interval specific inputs
        kwargs = args_to_kwargs(args, in_axes_outer.keys())
        max_length, starts, output = (
            kwargs[interval_max_length],
            kwargs[interval_starts],
            kwargs[output_name],
        )

        # runs batched computation
        kwargs[interval_max_length] = jnp.arange(max_length)
        output_interval = batched_function(*kwargs_to_args(kwargs))

        # set result in output
        indices_interval = starts[:, None] + jnp.arange(max_length)
        output = set_index_in_pytree(
            output, out_axes_outer, indices_interval, interval_axis, output_interval
        )
        return output

    outer_function = runtime_check_axis(outer_function, in_axes_outer, out_axes_outer)

    return set_documentation(outer_function, in_axes, out_axes, reference_func=f)
