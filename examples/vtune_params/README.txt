
This directory contains basic definitions for most of the VTune analysis modes.
In order to use/profile (from examples directory):

$ source ./vtune_params/concurrency.sh

--> defines environment variable "vrun" which the SLURM scripts use to specify the VTune command
--> defines alias "vfinal" which defines '-search-dir' option on all TOAST binary directorys
      and '-source-search-dir' on all TOAST source directories

Output in out_<script_name>/

$ vfinal <vtune_output_dir>

--> where <vtune_output_dir> will be something like out_<script_name>/vtune.nidXXXXX (where nidXXXXX is the job identifier)

NOTES:
  - Environment variable VTUNE_DATA_LIMIT can be set to over-ride the default of 0 (unlimited)
  - If TOAST is in a different location than [ ./vtune_params/../../ ], define TOAST_SOURCE_DIR in your environment
