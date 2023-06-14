#!/bin/bash

#1 folder to create redirections for
#2 folder to create redirection in
function create_redirection {

    for file in `find $1 -name '*html'`
    do 
        subdir="$(dirname "${file#$1/}")"
        redirection="${2:-.}/$subdir/$(basename $file)"
        if [ ! -f "$redirection" ]; then

            redirect_path="${2:+../}"
            for segment in "${subdir//\// }"
            do
                redirect_path+="../"
            done
            redirect_path+="$file"

            redirect_str='<html><head><meta http-equiv="refresh" content="0; url='
            redirect_str+="$redirect_path"
            redirect_str+='"></head></html>'

            echo "Redirecting $redirection to $file."
            mkdir -p "$(dirname $redirection)"
            echo $redirect_str > "$redirection"

            if $(git rev-parse --is-inside-work-tree 2> /dev/null); 
            then git add $redirection
            fi
        fi
    done    
}

versions=`echo */ | egrep -o "([0-9]{1,}\.)+[0-9]{1,}" | sort -r -V`
for version in $versions; 
do create_redirection $version latest
done
create_redirection latest
