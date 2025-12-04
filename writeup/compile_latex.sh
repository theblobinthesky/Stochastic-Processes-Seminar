#!/bin/bash

output_dir="../output"
root_file="root_document"

pdflatex -output-directory=$output_dir $root_file
makeglossaries -d $output_dir $root_file
pdflatex -output-directory=$output_dir $root_file
