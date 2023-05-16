# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import collections
import copy
import re
from collections import OrderedDict

import traitlets
from astropy import units as u
from traitlets import (
    Bool,
    Callable,
    Dict,
    Float,
    HasTraits,
    Instance,
    Int,
    List,
    Set,
    TraitError,
    TraitType,
    Tuple,
    Undefined,
    Unicode,
    UseEnum,
    signature_has_traits,
)

from .accelerator import ImplementationType, use_accel_jax, use_accel_omp
from .utils import Logger, import_from_name, object_fullname


def trait_string_to_scalar(val):
    """Attempt to convert a string to other basic python types.

    Trait containers support arbitrary objects, and there are situations where
    we just have a string value and need to determine the actual python type.
    This arises when parsing a config dictionary or when converting the config
    elements of a container.

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
    else:
        # See if we have a Quantity or Unit string representation
        try:
            parts = val.split()
            vstr = parts.pop(0)
            ustr = " ".join(parts)
            if vstr == "unit":
                # This string is a unit.  See if there is anything
                # following, and if not assume dimensionless_unscaled.
                if ustr == "":
                    return u.dimensionless_unscaled
                else:
                    return u.Unit(ustr)
            elif ustr == "":
                raise ValueError("Empty unit string")
            else:
                # See if we have a quantity
                value = float(vstr)
                unit = u.Unit(ustr)
                return u.Quantity(value, unit=unit)
        except (IndexError, ValueError):
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
                    return val


def trait_scalar_to_string(val):
    """Convert a scalar value into a string.

    This is needed to stringify both scalar traits and also container
    elements.

    Args:
        val (object):  A python scalar

    Returns:
        (str):  The string version.

    """
    if val is None:
        return "None"
    elif isinstance(val, u.UnitBase):
        return f"unit {str(val)}"
    elif isinstance(val, u.Quantity):
        return f"{val.value:0.14e} {str(val.unit)}"
    elif isinstance(val, bool):
        if val:
            return "True"
        else:
            return "False"
    elif isinstance(val, float):
        return f"{val:0.14e}"
    elif isinstance(val, int):
        return f"{val}"
    if isinstance(val, TraitConfig):
        # This trait value is a reference to another TraitConfig
        return "@config:{}".format(val.get_config_path())
    else:
        # Must be a string
        return val


# Add mixin methods to built-in Trait types

# Scalar base types

Bool.py_type = lambda self: bool
UseEnum.py_type = lambda self: self.enum_class
Float.py_type = lambda self: float
Int.py_type = lambda self: int
Unicode.py_type = lambda self: str


def _create_scalar_trait_get_conf(conf_type):
    # Create a class method that gets a config entry for a scalar
    # trait with no units.
    def _get_conf(self, obj=None):
        cf = dict()
        cf["type"] = conf_type
        if obj is None:
            v = self.default_value
        else:
            v = self.get(obj)
        cf["value"] = trait_scalar_to_string(v)
        cf["unit"] = "None"
        cf["help"] = str(self.help)
        return cf

    return _get_conf


Bool.get_conf = _create_scalar_trait_get_conf("bool")
UseEnum.get_conf = _create_scalar_trait_get_conf("enum")
Float.get_conf = _create_scalar_trait_get_conf("float")
Int.get_conf = _create_scalar_trait_get_conf("int")
Unicode.get_conf = _create_scalar_trait_get_conf("str")

# Container types.  These need specialized get_conf() methods.

List.py_type = lambda self: list


def list_get_conf(self, obj=None):
    cf = dict()
    cf["type"] = "list"
    if obj is None:
        val = self.default_value
    else:
        v = self.get(obj)
        if v is None:
            msg = (
                f"The toast config system does not support None values for "
                f"List traits. "
                f"Failed to parse '{self.name}' : '{self.help}'"
            )
            raise ValueError(msg)
            # val = "None"
        else:
            val = list()
            for item in v:
                val.append(trait_scalar_to_string(item))
    cf["value"] = val
    cf["unit"] = "None"
    cf["help"] = str(self.help)
    return cf


List.get_conf = list_get_conf

Set.py_type = lambda self: set


def set_get_conf(self, obj=None):
    cf = dict()
    cf["type"] = "set"
    if obj is None:
        val = self.default_value
    else:
        v = self.get(obj)
        if v is None:
            raise ValueError(
                "The toast config system does not support None values for Set traits."
            )
            # val = "None"
        else:
            val = set()
            for item in v:
                val.add(trait_scalar_to_string(item))
    cf["value"] = val
    cf["unit"] = "None"
    cf["help"] = str(self.help)
    return cf


Set.get_conf = set_get_conf

Dict.py_type = lambda self: dict


def dict_get_conf(self, obj=None):
    cf = dict()
    cf["type"] = "dict"
    if obj is None:
        val = self.default_value
    else:
        v = self.get(obj)
        if v is None:
            raise ValueError(
                "The toast config system does not support None values for Dict traits."
            )
            # val = "None"
        else:
            val = dict()
            for k, v in v.items():
                val[k] = trait_scalar_to_string(v)
    cf["value"] = val
    cf["unit"] = "None"
    cf["help"] = str(self.help)
    return cf


Dict.get_conf = dict_get_conf

Tuple.py_type = lambda self: tuple


def tuple_get_conf(self, obj=None):
    cf = dict()
    cf["type"] = "tuple"
    if obj is None:
        val = self.default_value
    else:
        v = self.get(obj)
        if v is None:
            raise ValueError(
                "The toast config system does not support None values for Tuple traits."
            )
            # val = "None"
        else:
            val = list()
            for item in v:
                val.append(trait_scalar_to_string(item))
            val = tuple(val)
    cf["value"] = val
    cf["unit"] = "None"
    cf["help"] = str(self.help)
    return cf


Tuple.get_conf = tuple_get_conf

Instance.py_type = lambda self: self.klass


def instance_get_conf(self, obj=None):
    cf = dict()
    cf["type"] = object_fullname(self.klass)
    if obj is None:
        val = self.default_value
    else:
        v = self.get(obj)
        if v is None:
            val = "None"
        elif isinstance(v, TraitConfig):
            val = trait_scalar_to_string(v)
        else:
            # There is nothing we can do with this
            val = "None"
    cf["value"] = val
    cf["unit"] = "None"
    cf["help"] = str(self.help)
    return cf


Instance.get_conf = instance_get_conf

Callable.py_type = lambda self: collections.abc.Callable


def callable_get_conf(self, obj=None):
    cf = dict()
    cf["type"] = "callable"
    if obj is None:
        val = self.default_value
    else:
        v = self.get(obj)
        if v is None:
            val = "None"
        else:
            # There is no way of serializing a generic callable
            # into a string.  Just set it to None.
            val = "None"
    cf["value"] = val
    cf["unit"] = "None"
    cf["help"] = str(self.help)
    return cf


Callable.get_conf = callable_get_conf


class Unit(TraitType):
    """A trait representing an astropy Unit."""

    default_value = u.dimensionless_unscaled
    info_text = "a Unit"

    def __init__(self, default_value=Undefined, **kwargs):
        super().__init__(default_value=default_value, **kwargs)

    def py_type(self):
        return u.Unit

    def get_conf(self, obj=None):
        cf = dict()
        cf["type"] = "Unit"
        if obj is None:
            val = self.default_value
        else:
            val = self.get(obj)
        cf["value"] = "unit"
        if val is None:
            cf["unit"] = "None"
        else:
            cf["unit"] = str(val)
        cf["help"] = str(self.help)
        return cf

    def validate(self, obj, value):
        if value is None:
            if self.allow_none:
                return None
            else:
                raise TraitError("Attempt to set trait to None, while allow_none=False")
        try:
            # Can we construct a unit from this?
            return u.Unit(value)
        except (ValueError, TypeError):
            msg = f"Value '{value}' can not be used to construct a Unit"
            raise TraitError(msg)

    def from_string(self, s):
        if self.allow_none and s == "None":
            return None
        return u.Unit(s)


class Quantity(Float):
    """A Quantity trait with units."""

    default_value = 0.0 * u.dimensionless_unscaled
    info_text = "a Quantity"

    def __init__(self, default_value=Undefined, **kwargs):
        super().__init__(default_value=default_value, **kwargs)

    def get_conf(self, obj=None):
        cf = dict()
        cf["type"] = "Quantity"
        if obj is None:
            v = self.default_value
        else:
            v = self.get(obj)
        if v is None:
            cf["value"] = "None"
            cf["unit"] = "None"
        else:
            cf["value"] = trait_scalar_to_string(v.value)
            cf["unit"] = str(v.unit)
        cf["help"] = str(self.help)
        return cf

    def validate(self, obj, value):
        if not isinstance(value, u.Quantity):
            # We can't read minds- force the user to specify the units
            msg = "Value '{}' does not have units".format(value)
            raise TraitError(msg)
        # Use the Float validation on the actual value
        valid_float = super().validate(obj, value.value)
        return u.Quantity(valid_float, value.unit)

    def from_string(self, s):
        if self.allow_none and s == "None":
            return None
        return u.Quantity(s)


def trait_docs(cls):
    """Decorator which adds trait properties to signature and docstring for a class.

    This appends a class docstring with argument help strings for every traitlet.  It
    also appends the traits to the constructor function signature.

    """
    doc = str(cls.__doc__)
    doc += "Note:\n"
    doc += "\t**The following traits can be set at construction or afterwards**\n\n"
    doc += "    Attributes:\n"
    # doc += "Attributes:\n"
    for trait_name, trait in cls.class_traits().items():
        cf = trait.get_conf()
        if cf["type"] == "Unit":
            default = cf["unit"]
        else:
            default = cf["value"]
        doc += "\t{} ({}):  {} (default = {})\n".format(
            trait_name, cf["type"], cf["help"], default
        )
    doc += "\n"
    cls.__doc__ = doc
    return signature_has_traits(cls)


class TraitConfig(HasTraits):
    """Base class for objects using traitlets and supporting configuration.

    This class implements some configuration functionality on top of the traitlets
    HasTraits base class.  The main features include:

        * Traitlet info and help string added to the docstring (cls.__doc__) for the
          class constructor.

        * Dump / Load of a named INSTANCE (not just a class) to a configuration file.
          This differs from the traitlets.Configuration package.

        * Creation and parsing of commandline options to set the traits on a named
          instance of the class.

    """

    name = Unicode(None, allow_none=True, help="The 'name' of this class instance")

    enabled = Bool(True, help="If True, this class instance is marked as enabled")

    kernel_implementation = UseEnum(
        ImplementationType,
        default_value=ImplementationType.DEFAULT,
        help="Which kernel implementation to use (DEFAULT, COMPILED, NUMPY, JAX).",
    )

    def __init__(self, **kwargs):
        self._accel_stack = list()
        super().__init__(**kwargs)
        if self.name is None:
            self.name = self.__class__.__qualname__

    def __repr__(self):
        val = "<{}".format(self.__class__.__qualname__)
        for trait_name, trait in self.traits().items():
            val += "\n  {} = {} # {}".format(trait_name, trait.get(self), trait.help)
        val += "\n>"
        return val

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
        ]

    def implementations(self):
        """Query which kernel implementations are supported.

        Returns:
            (list):  List of implementations.

        """
        return self._implementations()

    def _supports_accel(self):
        return False

    def supports_accel(self):
        """Query whether the operator supports accelerator kernels

        Returns:
            (bool):  True if the operator can use accelerators, else False.

        """
        return self._supports_accel()

    def select_kernels(self, use_accel=None):
        """Return the currently selected kernel implementation.

        This returns the kernel implementation that should be used

        Returns:
            (tuple):  The (ImplementationType, bool use_accel) switches.

        """
        impls = self.implementations()
        if use_accel is None:
            return ImplementationType.DEFAULT, False
        elif use_accel:
            if use_accel_jax:
                if ImplementationType.JAX not in impls:
                    msg = f"JAX accelerator use is enabled, "
                    msg += f"but not supported by {self.name}"
                    raise RuntimeError(msg)
                return ImplementationType.JAX, True
            else:
                if ImplementationType.COMPILED not in impls:
                    msg = f"OpenMP accelerator use is enabled, "
                    msg += f"but not supported by {self.name}"
                    raise RuntimeError(msg)
                return ImplementationType.COMPILED, True
        else:
            return ImplementationType.DEFAULT, False

    def __eq__(self, other):
        if len(self.traits()) != len(other.traits()):
            # print(
            #     f"DBG self has {len(self.traits())} traits, other has {len(other.traits())}"
            # )
            return False
        # Now we know that both objects have the same number of traits- compare the
        # types and values.
        for trait_name, trait in self.traits().items():
            trother = other.traits()[trait_name]
            if isinstance(trait, Set):
                tset = {x: x for x in trait.get(self)}
                oset = {x: x for x in trother.get(other)}
                if tset != oset:
                    # print(f"DBG {tset} != {oset}")
                    return False
            elif isinstance(trait, Dict):
                tdict = dict(trait.get(self))
                odict = dict(trother.get(other))
                if tdict != odict:
                    # print(f"DBG {tdict} != {odict}")
                    return False
            else:
                if trait.get(self) != trother.get(other):
                    # print(f"DBG trait {trait.get(self)} != {trother.get(other)}")
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def get_class_config_path(cls):
        return "/{}".format(cls.__qualname__)

    def get_config_path(self):
        if self.name is None:
            return None
        return "/{}".format(self.name)

    @staticmethod
    def _check_parent(conf, section, name):
        parent = conf
        if section is not None:
            path = section.split("/")
            for p in path:
                if p not in parent:
                    parent[p] = OrderedDict()
                parent = parent[p]
        if name in parent:
            msg = None
            if section is None:
                msg = "Config object {} already exists".format(name)
            else:
                msg = "Config object {}/{} already exists".format(section, name)
            raise TraitError(msg)
        return parent

    @classmethod
    def get_class_config(cls, section=None, input=None):
        """Return a dictionary of the default traits of a class.

        This returns a new or appended dictionary.  The class default properties are
        contained in a dictionary found in result[section][cls.name].  If the section
        string contains forward slashes, it is interpreted as a nested dictionary
        keys.  For example, if section == "sect1/sect2", then the resulting instance
        properties will be at result[sect1][sect2][cls.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            section (str):  The section to add properties to.
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        if input is None:
            input = OrderedDict()
        name = cls.__qualname__
        parent = cls._check_parent(input, section, name)
        parent[name] = OrderedDict()
        parent[name]["class"] = object_fullname(cls)
        for trait_name, trait in cls.class_traits().items():
            cf = trait.get_conf()
            parent[name][trait_name] = OrderedDict()
            parent[name][trait_name]["value"] = cf["value"]
            parent[name][trait_name]["unit"] = cf["unit"]
            parent[name][trait_name]["type"] = cf["type"]
            parent[name][trait_name]["help"] = cf["help"]
            # print(
            #     f"{name} class conf {trait_name}: {cf}"
            # )
        return input

    def get_config(self, section=None, input=None):
        """Return a dictionary of the current traits of a class *instance*.

        This returns a new or appended dictionary.  The class instance properties are
        contained in a dictionary found in result[section][self.name].  If the section
        string contains forward slashes, it is interpreted as a nested dictionary
        keys.  For example, if section == "sect1/sect2", then the resulting instance
        properties will be at result[sect1][sect2][self.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            section (str):  The section to add properties to.
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        if input is None:
            input = OrderedDict()
        name = self.name
        parent = self._check_parent(input, section, name)
        parent[name] = OrderedDict()
        parent[name]["class"] = object_fullname(self.__class__)
        for trait_name, trait in self.traits().items():
            cf = trait.get_conf(obj=self)
            parent[name][trait_name] = OrderedDict()
            parent[name][trait_name]["value"] = cf["value"]
            parent[name][trait_name]["unit"] = cf["unit"]
            parent[name][trait_name]["type"] = cf["type"]
            parent[name][trait_name]["help"] = cf["help"]
            # print(f"{name} instance conf {trait_name}: {cf}")
        return input

    @classmethod
    def translate(cls, props):
        """Translate config properties prior to construction.

        This method can be overridden by derived classes to provide a way of
        manipulating config properties prior to being parsed and passed to the
        constructor.  This is a way of detecting and accomodating old configuration
        information if the class code changes.

        Args:
            props (dict):  The original parameter information.

        Returns:
            (dict):  Modified parameters.

        """
        if "class" in props:
            del props["class"]
        return props

    @staticmethod
    def from_config(name, props):
        """Factory function to instantiate derived classes from a config.

        This function uses the 'class' key in the properties dictionary to instantiate
        the desired class and pass in the name and parameters to the constructor.

        Args:
            name (str):  The name of the class instance, passed to the constructor.
            props (dict):  This is a dictionary of properties corresponding to the
                format returned by the config() and class_config() methods.

        Returns:
            (TraitConfig):  The instantiated derived class.

        """
        if "class" not in props:
            msg = "Property dictionary does not contain 'class' key"
            raise RuntimeError(msg)
        cls_path = props["class"]
        cls = import_from_name(cls_path)

        # We got this far, so we have the class!  Perform any translation
        props = cls.translate(props)

        # Parse all the parameter type information and create values we will pass to
        # the constructor.
        kw = dict()
        kw["name"] = name
        for k, v in props.items():
            if v["type"] == "Unit":
                if v["value"] != "unit":
                    raise RuntimeError(
                        f"Unit trait does not have 'unit' as the conf value"
                    )
                if v["unit"] == "None":
                    kw[k] = None
                else:
                    kw[k] = u.Unit(v["unit"])
                # print(f"from_config {name}:    {k} = {kw[k]}")
            elif v["type"] == "Quantity":
                # print(f"from_config {name}:    {v}")
                if v["value"] == "None":
                    kw[k] = None
                else:
                    kw[k] = u.Quantity(float(v["value"]), u.Unit(v["unit"]))
                # print(f"from_config {name}:    {k} = {kw[k]}")
            elif v["type"] == "set":
                if v["value"] == "None":
                    kw[k] = None
                elif v["value"] == "{}":
                    kw[k] = set()
                else:
                    kw[k] = set([trait_string_to_scalar(x) for x in v["value"]])
                # print(f"from_config {name}:    {k} = {kw[k]}")
            elif v["type"] == "list":
                if v["value"] == "None":
                    kw[k] = None
                elif v["value"] == "[]":
                    kw[k] = list()
                else:
                    kw[k] = list([trait_string_to_scalar(x) for x in v["value"]])
                # print(f"from_config {name}:    {k} = {kw[k]}")
                continue
            elif v["type"] == "tuple":
                if v["value"] == "None":
                    kw[k] = None
                elif v["value"] == "()":
                    kw[k] = tuple()
                else:
                    kw[k] = tuple([trait_string_to_scalar(x) for x in v["value"]])
                # print(f"from_config {name}:    {k} = {kw[k]}")
            elif v["type"] == "dict":
                if v["value"] == "None":
                    kw[k] = None
                elif v["value"] == "{}":
                    kw[k] = dict()
                else:
                    # print(f"from_config input dict = {v['value']}")
                    kw[k] = {
                        x: trait_string_to_scalar(y) for x, y in v["value"].items()
                    }
                # print(f"from_config {name}:    {k} = {kw[k]}")
            elif v["value"] == "None":
                # Regardless of type, we set this to None
                kw[k] = None
            elif (
                v["type"] == "float"
                or v["type"] == "int"
                or v["type"] == "str"
                or v["type"] == "bool"
            ):
                kw[k] = trait_string_to_scalar(v["value"])
                # print(f"from_config {name}:    {k} = {kw[k]}")
            elif v["type"] == "unknown":
                # This was a None value in the TOML or similar unknown object
                pass
            else:
                # This is either a class instance of some arbitrary type,
                # or a callable.
                pass
        # Instantiate class and return
        return cls(**kw)
