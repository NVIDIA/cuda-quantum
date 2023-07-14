#!/bin/bash

SEARCH_FOLDER="/home/.local/lib/python3.10/site-packages/cudaq/"

# Get the size of each subdirectory within the wheel folder
echo "Getting the size of the entire cudaq directory (megabytes): "
total_size=$(du -s --block-size=1MB $SEARCH_FOLDER)
echo $total_size
echo ""
echo "Looping through the subdirectories and printing sizes in bytes: "
for f in $(find $SEARCH_FOLDER -type d); do 
  # Other options for `du` :
  # -a : list the size of each file as well
  # -s : displays a summary of the size of each directory
  printf "%-150s\n" "$(du -s $f)"
done 

# To further inspect folders manually, just run:
# echo $(du -a <specific/path>)
path_1="/home/.local/lib/python3.10/site-packages/cudaq/bin/"
echo ""
# echo $(du -a )
printf "%-150s\n" "$(du -a $path_1)"

path_2="/home/.local/lib/python3.10/site-packages/cudaq/lib/"
echo ""
# echo $(du -a )
printf "%-150s\n" "$(du -a $path_2)"