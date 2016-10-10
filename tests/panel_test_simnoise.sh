#!/bin/bash

panel="600x400"

dets="f1a f1b f2a f2b high white"

files=""
for d in ${dets}; do
    files="${files} out_test_simnoise_rawpsd_${d}.png"
    files="${files} out_test_simnoise_psd_${d}.png"
    files="${files} out_test_simnoise_tod_mc0_${d}.png"
    files="${files} out_test_simnoise_tod_var_${d}.png"
done

montage -tile 4x6 -geometry ${panel} ${files} simnoise.png


