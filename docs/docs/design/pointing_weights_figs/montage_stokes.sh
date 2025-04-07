#!/bin/bash

# To update figures:
# Copy the output hwp-static_* files from the ops_stokes_weights unit
# test output into this directory, and then run this script.

den=300
geom="576x576"

for hwp in "none" "0.0" "45.0" "90.0"; do
    infiles=""
    for case in "I1-Q0-U0" "I0-Q1-U0" "I0-Q0-U1" "I1-Q1-U1"; do
        infiles="${infiles} -density ${den} -geometry ${geom} hwp-static_${hwp}_test-0-0_${case}.pdf"
    done    
    montage -tile 4x1 ${infiles} "hwp_row_${hwp}.pdf"
done


