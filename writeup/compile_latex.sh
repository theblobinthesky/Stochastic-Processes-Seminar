#!/bin/bash

output_dir="../output"
root_file="root_document"

pdflatex -output-directory=$output_dir $root_file
cd $output_dir
bibtex $root_file
cd -
pdflatex -output-directory=$output_dir $root_file
