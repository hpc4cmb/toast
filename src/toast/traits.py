# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import collections
import copy
import re
import types
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
from .trait_utils import fix_quotes, string_to_trait, trait_to_string
from .utils import Logger, import_from_name, object_fullname

# Add mixin methods to built-in Trait types

# Scalar base types


def _create_scalar_trait_get_conf(conf_type, py_type):
    # Create a class method that gets a config entry for a scalar
    # trait with no units.
    def _get_conf(self, obj=None):
        cf = dict()
        cf["type"] = conf_type
        if obj is None:
            v = self.default_value
        else:
            v = self.get(obj)
        if v is None or v == traitlets.Undefined or v == traitlets.Sentinel:
            vpy = None
        elif py_type is None:
            vpy = v
        else:
            vpy = py_type(v)
        cf["value"] = trait_to_string(vpy)
        cf["help"] = str(self.help)
        return cf

    return _get_conf


Bool.get_conf = _create_scalar_trait_get_conf("bool", bool)
UseEnum.get_conf = _create_scalar_trait_get_conf("enum", None)
Float.get_conf = _create_scalar_trait_get_conf("float", float)
Int.get_conf = _create_scalar_trait_get_conf("int", int)
Unicode.get_conf = _create_scalar_trait_get_conf("str", str)

# Container types.


def _create_container_trait_get_conf(conf_type, py_type):
    # Create a class method that gets a config entry for a container
    # trait and checks for None.
    def _get_conf(self, obj=None):
        cf = dict()
        cf["type"] = conf_type
        if obj is None:
            v = self.default_value
        else:
            v = self.get(obj)
            if v is None:
                msg = "The toast config system does not support None values for "
                msg += f"{conf_type} traits. Failed to parse '{self.name}' :"
                msg += f" '{self.help}'"
                raise ValueError(msg)
        if v == traitlets.Undefined or v == traitlets.Sentinel:
            vpy = py_type()
        else:
            vpy = py_type(v)
        # print(f"DBG {conf_type} get_conf({v} -> {vpy})")
        cf["value"] = trait_to_string(vpy)
        cf["help"] = str(self.help)
        return cf

    return _get_conf


List.get_conf = _create_container_trait_get_conf("list", list)
Set.get_conf = _create_container_trait_get_conf("set", set)
Dict.get_conf = _create_container_trait_get_conf("dict", dict)
Tuple.get_conf = _create_container_trait_get_conf("tuple", tuple)


# Special case for Instance and Callable traits


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
            val = trait_to_string(v)
        else:
            # There is nothing we can do with this
            val = "None"
    cf["value"] = val
    cf["help"] = str(self.help)
    return cf


Instance.get_conf = instance_get_conf


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
            # into a string.  Just set it to None for now.
            val = "None"
    cf["value"] = val
    cf["help"] = str(self.help)
    return cf


Callable.get_conf = callable_get_conf

# Unit / Quantity traits


class Unit(TraitType):
    """A trait representing an astropy Unit."""

    default_value = u.dimensionless_unscaled
    info_text = "a Unit"

    def __init__(self, default_value=Undefined, **kwargs):
        super().__init__(default_value=default_value, **kwargs)

    def validate(self, obj, value):
        if value is None:
            if self.allow_none:
                return None
            else:
                raise TraitError(
                    f"Attempt to set trait {self.name} to None, while allow_none=False"
                )
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


Unit.get_conf = _create_scalar_trait_get_conf("Unit", u.Unit)


class Quantity(Float):
    """A Quantity trait with units."""

    default_value = 0.0 * u.dimensionless_unscaled
    info_text = "a Quantity"

    def __init__(self, default_value=Undefined, **kwargs):
        super().__init__(default_value=default_value, **kwargs)

    def validate(self, obj, value):
        if value is None:
            if self.allow_none:
                return None
            else:
                raise TraitError(
                    f"Attempt to set trait {self.name} to None, while allow_none=False"
                )
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


Quantity.get_conf = _create_scalar_trait_get_conf("Quantity", u.Quantity)


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

        * Dump / Load of a named INSTANCE (not just a class) to a configuration
          dictionary in memory.  This configuration dictionary serves as an
          intermediate representation which can then be translated into several
          configuration file formats.

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
            parent[name][trait_name]["type"] = cf["type"]
            parent[name][trait_name]["help"] = cf["help"]
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
            parent[name][trait_name]["type"] = cf["type"]
            parent[name][trait_name]["help"] = cf["help"]
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
        log = Logger.get()
        if "class" not in props:
            msg = "Property dictionary does not contain 'class' key"
            raise RuntimeError(msg)
        cls_path = props["class"]
        cls = import_from_name(cls_path)

        # We got this far, so we have the class!  Perform any translation
        props = cls.translate(props)

        # Parse all the parameter type information and create values we will pass to
        # the constructor.
        parsable = set(
            [
                "Unit",
                "Quantity",
                "set",
                "dict",
                "tuple",
                "list",
                "float",
                "int",
                "str",
                "bool",
                "enum",
            ]
        )
        kw = dict()
        kw["name"] = name
        avail_traits = set(cls.class_trait_names())
        for k, v in props.items():
            if k == "class":
                # print(f"CONF {name}: skipping class", flush=True)
                continue
            if k not in avail_traits:
                msg = f"Class {cls_path} currently has no configuration"
                msg += f" trait '{k}'.  This will be ignored, and your config "
                msg += "file is likely out of date."
                log.warning(msg)
                continue
            # print(f"CONF {name}: parsing {v}", flush=True)
            if isinstance(v["value"], str):
                if v["value"] == "None":
                    # Regardless of type, we set this to None
                    kw[k] = None
                    # print(f"CONF {name}:   (str -> None) {k} = None", flush=True)
                elif v["type"] == "unknown":
                    # This was a None value or similar unknown object
                    # print(f"CONF {name}:   (str, type unknown) {k} = pass", flush=True)
                    pass
                elif v["type"] in parsable:
                    kw[k] = string_to_trait(v["value"])
                    # print(f"CONF {name}:   (str -> parsable) {k} = {kw[k]}", flush=True)
                else:
                    # print(f"CONF {name}:   (str, nonparsable) {k} != {v['value']}", flush=True)
                    pass
            else:
                kw[k] = v["value"]
                # print(f"CONF {name}:   ({type(kw[k])}) {k} = {kw[k]}", flush=True)

        # Instantiate class and return
        # print(f"Instantiate class with {kw}", flush=True)
        return cls(**kw)


def create_from_config(conf):
    """Instantiate classes in a configuration.

    This iteratively instantiates classes defined in the configuration, replacing
    object names with references to those objects.  References to other objects in the
    config are specified with the string '@config:' followed by a UNIX-style "path"
    where each element of the path is a dictionary key in the config.  For example:

        @config:/operators/pointing

    Would reference an object at conf["operators"]["pointing"].  Object references like
    this only work if the target of the reference is a built-in type (str, float, int,
    etc) or a class derived from TraitConfig.

    Args:
        conf (dict):  the configuration

    Returns:
        (SimpleNamespace):  A namespace containing the sections and instantiated
            objects specified in the original config structure.

    """
    log = Logger.get()
    ref_prefix = "@config:"
    ref_pat = re.compile("^{}/(.*)".format(ref_prefix))

    def _get_node(path, tree):
        """Given the path as a list of keys, descend tree and return the node."""
        node = tree
        for elem in path:
            if isinstance(node, list):
                if not isinstance(elem, int):
                    msg = f"Node path {path}, element {elem} is not an integer."
                    msg += f" Cannot index list node {node}."
                    raise RuntimeError(msg)
                if elem >= len(node):
                    # We hit the end of our path, node does not yet exist.
                    return None
                node = node[elem]
            elif isinstance(node, dict):
                if elem not in node:
                    # We hit the end of our path, node does not yet exist.
                    return None
                node = node[elem]
            else:
                # We have hit a leaf node without getting to the end of our
                # path.  This means the node does not exist yet.
                return None
        return node

    def _dereference(value, tree):
        """If the object is a string with a path reference, return the object at
        that location in the tree.  Otherwise return the object.
        """
        if not isinstance(value, str):
            return value
        ref_mat = ref_pat.match(value)
        if ref_mat is not None:
            # This string contains a reference
            ref_path = ref_mat.group(1).split("/")
            return _get_node(ref_path, tree)
        else:
            # No reference, it is just a string
            return value

    def _insert_element(obj, parent_path, key, tree):
        """Insert object as a child of the parent path."""
        parent = _get_node(parent_path, tree)
        # print(f"{parent_path}: insert at {parent}: {obj}", flush=True)
        if parent is None:
            msg = f"Node {parent} at {parent_path} does not exist, cannot"
            msg += f" insert {obj}"
            raise RuntimeError(msg)
        elif isinstance(parent, list):
            parent.append(obj)
        elif isinstance(parent, dict):
            if key is None:
                msg = f"Node {parent} at {parent_path} cannot add {obj} without key"
                raise RuntimeError(msg)
            parent[key] = obj
        else:
            msg = f"Node {parent} at {parent_path} is not a list or dict"
            raise RuntimeError(msg)

    def _parse_string(value, parent_path, key, out):
        """Add a string to the output tree if it resolves."""
        obj = _dereference(value, out)
        # print(f"parse_string DEREF str {value} -> {obj}", flush=True)
        if obj is None:
            # Does not yet exist
            return 1
        # Add to output
        _insert_element(obj, parent_path, key, out)
        return 0

    def _parse_trait_value(obj, tree):
        """Recursively check trait value for references.
        Note that the input has already been tested for None values,
        so returning None from this function indicates that the
        trait value contains undefined references.
        """
        if isinstance(obj, str):
            temp = _dereference(obj, tree)
            # print(f"parse_trait DEREF str {obj} -> {temp}")
            return temp
        if isinstance(obj, list):
            ret = list()
            for it in obj:
                if it is None:
                    ret.append(None)
                else:
                    check = _parse_trait_value(it, tree)
                    if check is None:
                        return None
                    else:
                        ret.append(check)
            return ret
        if isinstance(obj, tuple):
            ret = list()
            for it in obj:
                if it is None:
                    ret.append(None)
                else:
                    check = _parse_trait_value(it, tree)
                    if check is None:
                        return None
                    else:
                        ret.append(check)
            return tuple(ret)
        if isinstance(obj, set):
            ret = set()
            for it in obj:
                if it is None:
                    ret.add(None)
                else:
                    check = _parse_trait_value(it, tree)
                    if check is None:
                        return None
                    else:
                        ret.add(check)
            return ret
        if isinstance(obj, dict):
            ret = dict()
            for k, v in obj.items():
                if v is None:
                    ret[k] = None
                else:
                    check = _parse_trait_value(v, tree)
                    if check is None:
                        return None
                    else:
                        ret[k] = check
            return ret
        # This must be some other scalar trait with no references
        return obj

    def _parse_traitconfig(value, parent_path, key, out, remaining):
        instance_name = None
        ctor = dict()
        for tname, tprops in value.items():
            if tname == "class":
                ctor["class"] = tprops
                continue
            ctor[tname] = dict()
            ctor[tname]["type"] = tprops["type"]
            tstring = tprops["value"]
            trait = string_to_trait(tstring)
            # print(f"{key} trait {tname} = {trait}", flush=True)
            if trait is None:
                ctor[tname]["value"] = None
                # print(f"{key} trait {tname} value = None", flush=True)
            else:
                check = _parse_trait_value(trait, out)
                # print(f"{key} trait {tname} value check = {check}", flush=True)
                if check is None:
                    # This trait contained unresolved references
                    # print(f"{key} trait {tname} value unresolved", flush=True)
                    remaining.add(tstring)
                    return 1
                ctor[tname]["value"] = check
        # If we got this far, it means that we parsed all traits and can
        # instantiate the class.
        # print(f"{parent_path}|{key}: parse_tc ctor = {ctor}", flush=True)
        obj = TraitConfig.from_config(instance_name, ctor)
        # print(f"{parent_path}|{key}: parse_tc {obj}", flush=True)
        _insert_element(obj, parent_path, key, out)
        return 0

    def _parse_list(value, parent_path, key, out, remaining):
        parent = _get_node(parent_path, out)
        # print(f"{parent_path}: parse_list parent = {parent}", flush=True)
        _insert_element(list(), parent_path, key, out)
        child_path = list(parent_path)
        child_path.append(key)
        # print(f"{parent_path}:   parse_list child = {child_path}", flush=True)
        unresolved = 0
        for val in value:
            if isinstance(val, list):
                unresolved += _parse_list(val, child_path, None, out, remaining)
                # print(f"parse_list: after {val} unresolved = {unresolved}", flush=True)
            elif isinstance(val, dict):
                if "class" in val:
                    # This is a TraitConfig instance
                    unresolved += _parse_traitconfig(val, child_path, None, out, remaining)
                    # print(
                    #     f"parse_list: after {val} unresolved = {unresolved}", flush=True
                    # )
                else:
                    # Just a normal dictionary
                    unresolved += _parse_dict(val, child_path, None, out, remaining)
                    # print(
                    #     f"parse_list: after {val} unresolved = {unresolved}", flush=True
                    # )
            else:
                unresolved += _parse_string(val, child_path, None, out)
                # print(f"parse_list: after {val} unresolved = {unresolved}", flush=True)
        return unresolved

    def _parse_dict(value, parent_path, key, out, remaining):
        parent = _get_node(parent_path, out)
        # print(f"{parent_path}: parse_dict parent = {parent}", flush=True)
        _insert_element(OrderedDict(), parent_path, key, out)
        child_path = list(parent_path)
        child_path.append(key)
        # print(f"{parent_path}:   parse_dict child = {child_path}", flush=True)
        unresolved = 0
        for k, val in value.items():
            if isinstance(val, list):
                unresolved += _parse_list(val, child_path, k, out, remaining)
                # print(f"parse_dict: after {k} unresolved = {unresolved}", flush=True)
            elif isinstance(val, dict):
                if "class" in val:
                    # This is a TraitConfig instance
                    unresolved += _parse_traitconfig(val, child_path, k, out, remaining)
                    # print(
                    #     f"parse_dict: after {k} unresolved = {unresolved}", flush=True
                    # )
                else:
                    # Just a normal dictionary
                    unresolved += _parse_dict(val, child_path, k, out, remaining)
                    # print(
                    #     f"parse_dict: after {k} unresolved = {unresolved}", flush=True
                    # )
            else:
                unresolved += _parse_string(val, child_path, k, out)
                # print(f"parse_dict: after {k} unresolved = {unresolved}", flush=True)
        return unresolved

    # Iteratively instantiate objects

    out = OrderedDict()
    done = False
    last_unresolved = None

    it = 0
    while not done:
        # print("PARSE iter ", it)
        done = True
        unresolved = 0
        remaining = set()

        # Go through the top-level dictionary
        for topkey in list(conf.keys()):
            if isinstance(conf[topkey], str):
                out[topkey] = conf[topkey]
                continue
            if len(conf[topkey]) > 0:
                unresolved += _parse_dict(conf[topkey], list(), topkey, out, remaining)
                # print(f"PARSE {it}: {topkey} unresolved now {unresolved}", flush=True)

        if last_unresolved is not None:
            if unresolved == last_unresolved:
                msg = f"Cannot resolve all references ({unresolved} remaining)"
                msg += f" in the configuration:  {remaining}"
                log.error(msg)
                raise RuntimeError(msg)
        last_unresolved = unresolved
        if unresolved > 0:
            done = False
        it += 1

    # Convert this recursively into a namespace for easy use
    root_temp = dict()
    for sect in list(out.keys()):
        if isinstance(out[sect], str):
            root_temp[sect] = out[sect]
        else:
            sect_ns = types.SimpleNamespace(**out[sect])
            root_temp[sect] = sect_ns
    out_ns = types.SimpleNamespace(**root_temp)
    return out_ns
