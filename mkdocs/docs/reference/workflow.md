# Workflows

TOAST "workflows" are just python scripts which use some helper functions for setting up one or more `Operator` instances using configuration information from external files or the commandline.  Not every python script using TOAST needs to make use of these tools and be a formal "workflow".  For example, one can write a python script which instantiates operators as you need them with all parameters specified in the script and no information supplied from the command line.  Using the workflow tools are most useful when a standard simulation / processing sequence will be run on multiple datasets with different options.


