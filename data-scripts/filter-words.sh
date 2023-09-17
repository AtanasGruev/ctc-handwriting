#!/bin/bash

if [ $# -eq 2 ] && [ -f ${1} ]; then
    less ${1} | grep -v '#' | awk -F' ' '{print $1, $2, $NF}' > ${2}
else
    echo "Usage: ./filter-words.sh src-file dst-file"
fi
