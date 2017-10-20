#!/bin/bash

#set -o errexit

: ${TOAST_SOURCE_DIR:="$(readlink -f $(realpath $(dirname ${BASH_SOURCE[0]})/../..))"}
export TOAST_SOURCE_DIR

echo -e "Using TOAST_SOURCE_DIR = \"${TOAST_SOURCE_DIR}\"..."

if [ ! -d "${TOAST_SOURCE_DIR}" ]; then
    echo -e "TOAST_SOURCE_DIR (${TOAST_SOURCE_DIR}) does not exist"
    return 1
fi

#------------------------------------------------------------------------------#
add-search-dir()
{               
    for i in $@ 
    do
        if [ ! -d "${i}" ]; then
            continue
        fi
        i="$(readlink -f $(realpath ${i}))"
        if [ -z "${SEARCH_DIRS}" ]; then 
            SEARCH_DIRS="-search-dir ${i}"
        else                              
            SEARCH_DIRS="${SEARCH_DIRS} -search-dir $i"
        fi                                             
    done                                               
}                                                      

#------------------------------------------------------------------------------#
add-source-search-dir()
{                      
    for i in $@        
    do           
        if [ ! -d "${i}" ]; then
            continue
        fi
        i="$(readlink -f $(realpath ${i}))"
        if [ -z "${SOURCE_SEARCH_DIRS}" ]; then 
            SOURCE_SEARCH_DIRS="-source-search-dir ${i}"
        else                                            
            SOURCE_SEARCH_DIRS="${SOURCE_SEARCH_DIRS} -source-search-dir $i"
        fi                                                                  
    done                                                                    
}                                                                           

#------------------------------------------------------------------------------#
# add toast bin directories to symbol search
for i in $(echo ${PATH} | sed 's/:/ /g')
do
    echo "${i}" | grep -i "toast" &> /dev/null
    RET="$?"
    if [ "${RET}" -eq 0 ]; then
        add-search-dir ${i}
    fi
done

#------------------------------------------------------------------------------#
# add LD_LIBRARY_PATH directories to symbol search
for i in $(echo ${LD_LIBRARY_PATH} | sed 's/:/ /g')
do
    add-search-dir ${i}
done


#------------------------------------------------------------------------------#
# add TOAST_SOURCE_DIR
for i in $(find ${TOAST_SOURCE_DIR}/src -type d | egrep -v '\.libs|\.deps|/gtest')
do
    DIR=$(readlink -f $(realpath ${i}))
    if [ -d "${DIR}" ]; then
        add-source-search-dir ${DIR}
    else
        echo -e "Directory \"${DIR}\" does not exist"
    fi
done

export _VTUNE_SEARCH_DIRS="${SEARCH_DIRS}"
export _VTUNE_SOURCE_DIRS="${SOURCE_SEARCH_DIRS}"
unset SEARCH_DIRS
unset SOURCE_SEARCH_DIRS

#------------------------------------------------------------------------------#

alias vfinal="amplxe-cl -finalize ${_VTUNE_SEARCH_DIRS} ${_VTUNE_SOURCE_DIRS} -r"
