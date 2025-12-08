#!/bin/bash

output_dir="../output-writeup"
root_file="writeup"

pdflatex -output-directory=$output_dir $root_file
cd $output_dir
bibtex $root_file
cd -
pdflatex -output-directory=$output_dir $root_file
