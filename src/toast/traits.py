# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import importlib

from collections import OrderedDict

import traitlets

from traitlets import (
    signature_has_traits,
    HasTraits,
    TraitError,
    Undefined,
    Unicode,
    Bool,
    List,
    Set,
    Dict,
    Tuple,
    Instance,
    Int,
    Float,
)

from astropy import units as u


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


def object_fullname(o):
    """Return the fully qualified name of an object."""
    module = o.__module__
    if module is None or module == str.__module__:
        return o.__qualname__
    return "{}.{}".format(module, o.__qualname__)


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

    @staticmethod
    def _check_parent(conf, section, name):
        parent = conf
        if section is not None:
            path = section.split("/")
            for p in path:
                if p not in parent:
                    parent[p] = dict()
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
    def _format_conf_trait(trt, tval):
        valstr = "None"
        unitstr = "None"
        typestr = None
        if isinstance(trt, Quantity):
            if tval is not None:
                valstr = "{:0.14e}".format(tval.value)
                unitstr = str(tval.unit)
        else:
            if tval is not None:
                if isinstance(trt, Float):
                    valstr = "{:0.14e}".format(tval)
                else:
                    valstr = "{}".format(tval)
        typestr = trait_type_to_string(trt)
        return valstr, unitstr, typestr

    @classmethod
    def class_config(cls, section=None, input=None):
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
            valstr, unitstr, typestr = cls._format_conf_trait(trait, trdefault)
            parent[name][trname]["value"] = valstr
            parent[name][trname]["unit"] = unitstr
            parent[name][trname]["type"] = typestr
            parent[name][trname]["help"] = trhelp
        return input

    def config(self, section=None, input=None):
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
            valstr, unitstr, typestr = self._format_conf_trait(trait, trval)
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
        cls_parts = cls_path.split(".")
        cls_name = cls_parts.pop()
        cls_mod_name = ".".join(cls_parts)
        cls = None
        try:
            cls_mod = importlib.import_module(cls_mod_name)
            cls = getattr(cls_mod, cls_name)
        except:
            msg = "Cannot import class '{}' from module '{}'".format(
                cls_name, cls_mod_name
            )
            raise RuntimeError(msg)
        # We got this far, so we have the class!  Perform any translation
        original = copy.deepcopy(props)
        props = cls.translate(original)

        # Parse all the parameter type information and create values we will pass to
        # the constructor.
        kw = dict()
        kw["name"] = name
        for k, v in props.items():
            if v["unit"] == "None":
                # Normal scalar, no units
                if v["value"] == "None":
                    kw[k] = None
                else:
                    pyt = string_to_pytype(v["type"])
                    if pyt is None:
                        # This is some kind of more complicated class.  We will let the
                        # constructor choose the default value.
                        continue
                    kw[k] = pyt(v["value"])
            else:
                # We have a Quantity.
                kw[k] = u.Quantity(float(v["value"]) * u.Unit(v["unit"]))
        # Instantiate class and return
        return cls(**kw)


def build_config(objects):
    """Build a configuration of current values.

    Args:
        objects (list):  A list of class instances to add to the config.  These objects
            must inherit from the TraitConfig base class.

    Returns:
        (dict):  The configuration.

    """
    conf = OrderedDict()
    for o in objects:
        conf = o.config(input=conf)
    return conf


def add_config_args(parser, conf, section, ignore=list(), prefix="", separator=":"):
    """Add arguments to an argparser for each parameter in a config dictionary.

    Using a previously created config dictionary, add a commandline argument for each
    object parameter in a section of the config.  The type, units, and help string for
    the commandline argument come from the config, which is in turn built from the
    class traits of the object.  Boolean parameters are converted to store_true or
    store_false actions depending on their current value.

    Args:
        parser (ArgumentParser):  The parser to append to.
        conf (dict):  The configuration dictionary.
        section (str):  Process objects in this section of the config.
        ignore (list):  List of object parameters to ignore when adding args.
        prefix (str):  Prepend this to the beginning of all options.
        separator (str):  Use this character between the class name and parameter.

    Returns:
        None

    """
    parent = conf
    if section is not None:
        path = section.split("/")
        for p in path:
            if p not in parent:
                msg = "section {} does not exist in config".format(section)
                raise RuntimeError(msg)
            parent = parent[p]
    for obj, props in parent.items():
        for name, info in props.items():
            if name in ignore:
                # Skip this as requested
                continue
            if name == "class":
                # This is not a user-configurable parameter.
                continue
            if info["type"] not in [bool, int, float, str, u.Quantity]:
                # This is not something that we can get from parsing commandline
                # options.  Skip it.
                continue
            if info["type"] is bool:
                # special case for boolean
                option = "--{}{}{}{}".format(prefix, obj, separator, name)
                act = "store_true"
                if info["value"]:
                    act = "store_false"
                    option = "--{}{}{}no_{}".format(prefix, obj, separator, name)
                parser.add_argument(
                    option,
                    required=False,
                    default=info["value"],
                    action=act,
                    help=info["help"],
                )
            else:
                option = "--{}{}{}{}".format(prefix, obj, separator, name)
                parser.add_argument(
                    option,
                    required=False,
                    default=info["value"],
                    type=info["type"],
                    help=info["help"],
                )
    return


def args_update_config(args, conf, defaults, section, prefix="", separator=":"):
    """Override options in a config dictionary from args namespace.

    Args:
        args (namespace):  The args namespace returned by ArgumentParser.parse_args()
        conf (dict):  The configuration to update.
        defaults (dict):  The starting default config, used to detect which options from
            argparse have been changed by the user.
        section (str):  Process objects in this section of the config.
        prefix (str):  Prepend this to the beginning of all options.
        separator (str):  Use this character between the class name and parameter.

    Returns:
        (namespace):  The un-parsed remaining arg vars.

    """
    remain = copy.deepcopy(args)
    parent = conf
    dparent = defaults
    if section is not None:
        path = section.split("/")
        for p in path:
            if p not in parent:
                msg = "section {} does not exist in config".format(section)
                raise RuntimeError(msg)
            parent = parent[p]
        for p in path:
            if p not in dparent:
                msg = "section {} does not exist in defaults".format(section)
                raise RuntimeError(msg)
            dparent = dparent[p]
    # Build the regex match of option names
    obj_pat = re.compile("{}(.*?){}(.*)".format(prefix, separator))
    for arg in vars(args):
        val = getattr(args, arg)
        obj_mat = obj_pat.match(arg)
        if obj_mat is not None:
            name = obj_mat.group(1)
            optname = obj_mat.group(2)
            if name not in parent:
                msg = (
                    "Parsing option '{}', config does not have object named {}".format(
                        arg, name
                    )
                )
                raise RuntimeError(msg)
            if name not in dparent:
                msg = "Parsing option '{}', defaults does not have object named {}".format(
                    arg, name
                )
                raise RuntimeError(msg)
            # Only update config options which are different than the default.
            # Otherwise we would be overwriting values from any config files with the
            # defaults from argparse.
            if val != dparent[name][optname]:
                parent[name][optname] = val
            # This arg was recognized, remove from the namespace.
            del remain.arg
    return remain
