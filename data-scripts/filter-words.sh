#!/bin/bash

BAD_SAMPLES=('a01-117-05-02' 'r06-022-03-05') # known bad samples in IAM dataset
# tmpfile=$(mktemp)

if [ $# -eq 2 ] && [ -f ${1} ]; then
    # Collect filename, transcription status, binarization threshold and transcription
    less ${1} | grep -v '#' | awk -F' ' '{print $1, $2, $3, $NF}' > ${2}
    
    # Remove knowm bad samples in-place for the dst-fle
    for sample in ${BAD_SAMPLES[@]}; do 
        sed -i "/^${sample}/d" ${2};
    done 
else
    echo "Usage: ./filter-words.sh src-file dst-file"
fi
