#!/bin/bash

command=$1
if [ "x${command}" = "x" ]; then
    echo "Usage:  $0 <command to run with --help argument>"
    echo "    Result is dumped to stdout for redirection."
    exit 0
fi

out='```{code-block} console
'

commout=$(eval ${command} --help 2>&1)

echo "${out}"
echo "${commout}"
echo ""
echo '```'
