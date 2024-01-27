# Copyright (c) 2023-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re

import astropy.units as u
from astropy.units import Quantity, Unit

from .utils import Logger


def fix_quotes(s, force=False):
    clean = s.strip(" '\"")
    if len(s) == 0 or force:
        return f"'{clean}'"
    else:
        return clean


def string_to_scalar(val):
    """Attempt to convert a string to supported scalar types.

    This handles the special case of Quantities and Units expressed as a string
    with a space separating the value and unit.

    Args:
        val (str):  The input.

    Returns:
        (scalar):  The converted value.

    """
    if not isinstance(val, str):
        # This is an already-instantiated object
        return val
    if val == "None":
        return None
    elif val == "True":
        return True
    elif val == "False":
        return False
    elif val == "":
        return val
    elif re.match(r"^Quantity.*", val) is not None:
        return eval(val)
    elif re.match(r"^Unit.*", val) is not None:
        return eval(val)
    else:
        # See if we have a legacy Quantity or Unit string representation.
        # Remove next few lines after sufficient deprecation period.
        try:
            qval = convert_legacy_quantity(val)
            return qval
        except ValueError:
            # No.  Try int next
            try:
                ival = int(val)
                # Yes
                return ival
            except ValueError:
                # No.  Try float
                try:
                    fval = float(val)
                    # Yes
                    return fval
                except ValueError:
                    # String or some other object
                    return fix_quotes(val)


def scalar_to_string(val, force=False):
    """Convert a scalar value into a string.

    This converts the value into a string representation which can be
    reversed with the `eval()` function.

    Args:
        val (object):  A python scalar

    Returns:
        (str):  The string version.

    """
    if val is None:
        return "None"
    elif isinstance(val, u.UnitBase):
        return f"Unit('{str(val)}')"
    elif isinstance(val, u.Quantity):
        return f"Quantity('{val.value:0.14e} {str(val.unit)}')"
    elif isinstance(val, bool):
        if val:
            return "True"
        else:
            return "False"
    elif isinstance(val, float):
        return f"{val:0.14e}"
    elif isinstance(val, int):
        return f"{val}"
    elif hasattr(val, "name") and hasattr(val, "enabled"):
        # This trait value is a reference to another TraitConfig
        return f"'@config:{val.get_config_path()}'"
    else:
        # Must be a string or other object
        if isinstance(val, str):
            val = fix_quotes(val, force=force)
        return val


def string_to_trait(val):
    """Attempt to convert a string to an arbitrary trait.

    Args:
        val (str):  The input.

    Returns:
        (scalar):  The converted value.

    """
    list_pat = re.compile(r"\[.*\]")
    set_pat = re.compile(r"\{.*\}")
    set_alt_pat = re.compile(r"set\(.*\)")
    tuple_pat = re.compile(r"\(.*\)")
    dict_pat = re.compile(r"\{.*:.*\}")
    if not isinstance(val, str):
        # This is an already-instantiated object
        return val
    bareval = fix_quotes(val)
    if (
        list_pat.match(bareval) is not None
        or set_pat.match(bareval) is not None
        or set_alt_pat.match(bareval) is not None
        or dict_pat.match(bareval) is not None
        or tuple_pat.match(bareval) is not None
    ):
        # print(f"DBG calling eval on container {bareval}", flush=True)
        # The string is a container.  Just eval it.
        container = eval(bareval)
        # FIXME:  Remove this call after sufficient deprecation period.
        return parse_deprecated_quantities(container)
    # It must be a scalar
    # print(f"DBG calling string_to_scalar {bareval}", flush=True)
    return string_to_scalar(bareval)


def trait_to_string(val):
    """Convert a trait into a string.

    This creates a string which can be passed to `eval()` to re-create
    the original container.

    Args:
        val (object):  A python scalar

    Returns:
        (str):  The string version.

    """

    def _convert_elem(v, nest):
        if isinstance(v, dict):
            s = _convert_dict(v, nest + 1)
        elif isinstance(v, set):
            s = _convert_set(v, nest + 1)
        elif isinstance(v, list):
            s = _convert_list(v, nest + 1)
        elif isinstance(v, tuple):
            s = _convert_tuple(v, nest + 1)
        else:
            s = scalar_to_string(v, force=(nest > 0))
        return s

    def _convert_dict(t, nest):
        out = "{"
        for k, v in t.items():
            s = _convert_elem(v, nest + 1)
            out += f"'{k}':{s},"
        out += "}"
        return out

    def _convert_set(t, nest):
        if len(t) == 0:
            return "set()"
        out = "{"
        for v in t:
            s = _convert_elem(v, nest + 1)
            out += f"{s},"
        out += "}"
        return out

    def _convert_list(t, nest):
        out = "["
        for v in t:
            s = _convert_elem(v, nest + 1)
            out += f"{s},"
        out += "]"
        return out

    def _convert_tuple(t, nest):
        out = "("
        for v in t:
            s = _convert_elem(v, nest + 1)
            out += f"{s},"
        out += ")"
        return out

    out = _convert_elem(val, 0)
    # print(
    #     f"DBG converted {val} to str '{out}'",
    #     flush=True,
    # )
    return out


def convert_legacy_quantity(qstring):
    """Convert and return old-style quantity string."""
    log = Logger.get()
    try:
        parts = qstring.split()
        vstr = parts.pop(0)
        ustr = " ".join(parts)
        if vstr == "unit":
            # This string is a unit.  See if there is anything
            # following, and if not assume dimensionless_unscaled.
            if ustr == "" or ustr == "None":
                out = u.dimensionless_unscaled
            else:
                out = u.Unit(ustr)
        elif ustr == "":
            raise ValueError("Empty unit string")
        else:
            # See if we have a quantity
            value = float(vstr)
            unit = u.Unit(ustr)
            out = u.Quantity(value, unit=unit)
        # We have one of these, raise warning
        if isinstance(out, u.Unit):
            msg = f"Legacy Unit string '{qstring}' is deprecated. "
            msg += f"Use 'Unit(\"{ustr}\")' instead."
        else:
            msg = f"Legacy Quantity string '{qstring}' is deprecated. "
            msg += f"Use 'Quantity(\"{qstring}\")' instead."
        log.warning(msg)
        return out
    except (IndexError, ValueError, TypeError):
        # Nope, not a legacy quantity string
        raise ValueError("Not a legacy quantity string")


def parse_deprecated_quantities(container):
    """Attempt to parse container values with deprecated Quantity strings.

    Old config files stored Quantities as a string with just the value and
    units (rather than a string which can be eval'd directly into the object).
    This function attempts to handle that case and also print a warning.

    Args:
        container (object):  One of the supported containers

    Returns:
        (object):  The input container with quantity strings instantiated.

    """

    def _parse_obj(c):
        if isinstance(c, list):
            return _parse_list(c)
        elif isinstance(c, tuple):
            return _parse_tuple(c)
        elif isinstance(c, set):
            return _parse_set(c)
        elif isinstance(c, dict):
            return _parse_dict(c)
        else:
            if isinstance(c, str):
                try:
                    out = convert_legacy_quantity(c)
                    return out
                except ValueError:
                    return c
            else:
                return c

    def _parse_list(c):
        out = list()
        for it in c:
            out.append(_parse_obj(it))
        return out

    def _parse_tuple(c):
        out = list()
        for it in c:
            out.append(_parse_obj(it))
        return tuple(out)

    def _parse_set(c):
        out = set()
        for it in c:
            out.add(_parse_obj(it))
        return out

    def _parse_dict(c):
        out = dict()
        for k, v in c.items():
            out[k] = _parse_obj(v)
        return out

    return _parse_obj(container)
