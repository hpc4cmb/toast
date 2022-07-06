# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

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
    Tuple,
    Undefined,
    Unicode,
    signature_has_traits,
)

from .utils import import_from_name, object_fullname


class Quantity(Float):
    """A Quantity trait with units."""

    default_value = 0.0 * u.dimensionless_unscaled
    info_text = "a Quantity"

    def __init__(self, default_value=Undefined, **kwargs):
        super().__init__(default_value=default_value, **kwargs)

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


def trait_type_to_string(trait):
    """Return a python type name corresponding to a trait.

    For the specified traitlet type, return the string name of the python type that
    should be used when assigning to the trait.

    Args:
        trait (traitlet.TraitType):  The trait.

    Returns:
        (str):  The string name.

    """
    if isinstance(trait, Bool):
        return "bool"
    elif isinstance(trait, List):
        return "list"
    elif isinstance(trait, Set):
        return "set"
    elif isinstance(trait, Dict):
        return "dict"
    elif isinstance(trait, Tuple):
        return "tuple"
    elif isinstance(trait, Quantity):
        return "Quantity"
    elif isinstance(trait, Float):
        return "float"
    elif isinstance(trait, Int):
        return "int"
    elif isinstance(trait, Instance):
        return trait.klass.__qualname__
    return "str"


def trait_string_to_value(val):
    """Attempt to convert a string to other basic types.

    Trait containers support arbitrary objects, but when parsing a config everything
    is a string.  This attempts to convert a value to its real type.

    Args:
        val (str):  The input.

    Returns:
        (scalar):  The converted value.

    """
    if isinstance(val, TraitConfig):
        # Reference to another object
        return val
    elif val == "None":
        return None
    elif val == "True":
        return True
    elif val == "False":
        return False
    elif val == "":
        return val
    else:
        # See if we have a Quantity string representation
        try:
            parts = val.split()
            vstr = parts.pop(0)
            ustr = " ".join(parts)
            if ustr == "":
                raise ValueError("No unit")
            v = float(vstr)
            unit = u.Unit(ustr)
            # Yes
            return u.Quantity(v, unit=unit)
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
                    # Just a string
                    return val


def string_to_pytype(st):
    """Return a python type corresponding to a type string.

    Used for parsing config properties.

    Args:
        st (str):  The type name.

    Returns:
        (class):  The python type.

    """
    if st == "bool":
        return bool
    elif st == "list":
        return list
    elif st == "set":
        return set
    elif st == "dict":
        return dict
    elif st == "tuple":
        return tuple
    elif st == "Quantity":
        return u.Quantity
    elif st == "int":
        return int
    elif st == "float":
        return float
    elif st == "str":
        return str
    # Must be a custom class...
    return None


def trait_info(trait):
    """Extract the trait properties.

    Returns:
        (tuple):  The name, python type, default value, and help string.

    """
    trtype = str
    if isinstance(trait, Bool):
        trtype = bool
    elif isinstance(trait, List):
        trtype = list
    elif isinstance(trait, Set):
        trtype = set
    elif isinstance(trait, Dict):
        trtype = dict
    elif isinstance(trait, Tuple):
        trtype = tuple
    elif isinstance(trait, Quantity):
        trtype = u.Quantity
    elif isinstance(trait, Float):
        trtype = float
    elif isinstance(trait, Int):
        trtype = int
    elif isinstance(trait, Instance):
        trtype = trait.klass
    return (trait.name, trtype, trait.default_value, trait.help)


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
        default = trait.default_value
        trait_type = trait_type_to_string(trait)
        if trait_type == "str":
            default = "'{}'".format(default)
        doc += "\t{} ({}):  {} (default = {})\n".format(
            trait_name, trait_type, trait.help, default
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.name is None:
            self.name = self.__class__.__qualname__

    def __repr__(self):
        val = "<{}".format(self.__class__.__qualname__)
        for trait_name, trait in self.traits().items():
            val += "\n  {} = {} # {}".format(trait_name, trait.get(self), trait.help)
        val += "\n>"
        return val

    def __eq__(self, other):
        if len(self.traits()) != len(other.traits()):
            return False
        # Now we know that both objects have the same number of traits- compare the
        # types and values.
        for trait_name, trait in self.traits().items():
            trother = other.traits()[trait_name]
            if isinstance(trait, Set):
                tset = {x: x for x in trait.get(self)}
                oset = {x: x for x in trother.get(other)}
                if tset != oset:
                    return False
            elif isinstance(trait, Dict):
                tdict = dict(trait.get(self))
                odict = dict(trother.get(other))
                if tdict != odict:
                    return False
            else:
                if trait.get(self) != trother.get(other):
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

    @staticmethod
    def _format_conf_trait(conf, trt, tval):
        retval = "None"
        unitstr = "None"
        typestr = trait_type_to_string(trt)

        def _format_item(c, tv):
            val = "None"
            unit = "None"
            if tv is None:
                return (val, unit)
            if isinstance(tv, TraitConfig):
                # We are dumping an instance which has handles to other TraitConfig
                # classes.  Do this recursively.
                c = tv.get_config(input=c)
                val = "@config:{}".format(tv.get_config_path())
            elif isinstance(tv, u.Quantity):
                val = "{:0.14e}".format(tv.value)
                unit = str(tv.unit)
            elif isinstance(tv, float):
                val = "{:0.14e}".format(tv)
            else:
                val = "{}".format(tv)
            return (val, unit)

        if isinstance(trt, Dict):
            if tval is not None:
                retval = dict()
                for k, v in tval.items():
                    vstr, vunit = _format_item(conf, v)
                    if vunit == "None":
                        retval[k] = vstr
                    else:
                        retval[k] = f"{vstr} {vunit}"
        elif isinstance(trt, List) or isinstance(trt, Set) or isinstance(trt, Tuple):
            if tval is not None:
                retval = list()
                for v in tval:
                    vstr, vunit = _format_item(conf, v)
                    if vunit == "None":
                        retval.append(vstr)
                    else:
                        retval.append(f"{vstr} {vunit}")
        elif isinstance(trt, Instance) and not isinstance(tval, TraitConfig):
            # Our trait is some other class not derived from TraitConfig.  This
            # means that we cannot recursively dump it to the config and we also have
            # no way (currently) of serializing this instance to the config.  We set it
            # to None so that default actions can be taken by the constructor.
            pass
        else:
            # Single object
            retval, unitstr = _format_item(conf, tval)

        return retval, unitstr, typestr

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
            trname, trtype, trdefault, trhelp = trait_info(trait)
            parent[name][trname] = OrderedDict()
            valstr, unitstr, typestr = cls._format_conf_trait(input, trait, trdefault)
            parent[name][trname]["value"] = valstr
            parent[name][trname]["unit"] = unitstr
            parent[name][trname]["type"] = typestr
            parent[name][trname]["help"] = trhelp
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
            trname, trtype, trdefault, trhelp = trait_info(trait)
            trval = None
            if trait.get(self) is not None:
                try:
                    trval = trtype(trait.get(self))
                except Exception:
                    trval = str(trait.get(self))
            parent[name][trname] = OrderedDict()
            valstr, unitstr, typestr = self._format_conf_trait(input, trait, trval)
            parent[name][trname]["value"] = valstr
            parent[name][trname]["unit"] = unitstr
            parent[name][trname]["type"] = typestr
            parent[name][trname]["help"] = trhelp
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
            if v["unit"] != "None":
                # Scalar Quantity
                kw[k] = u.Quantity(float(v["value"]) * u.Unit(v["unit"]))
                continue
            if v["value"] == "None":
                # None value
                kw[k] = None
                continue
            pyt = string_to_pytype(v["type"])
            if pyt is None:
                # This is some kind of more complicated class.  We will let the
                # constructor choose the default value.
                continue
            if v["value"] == "{}" or v["value"] == "()" or v["value"] == "[]":
                # Empty container
                kw[k] = pyt()
                continue
            if pyt == list or pyt == set or pyt == tuple:
                kw[k] = pyt([trait_string_to_value(x) for x in v["value"]])
                continue
            if pyt == dict:
                # Convert items
                kw[k] = dict()
                for dk, dv in v["value"].items():
                    kw[k][dk] = trait_string_to_value(dv)
                continue
            # Other scalar
            kw[k] = trait_string_to_value(v["value"])
        # Instantiate class and return
        return cls(**kw)
