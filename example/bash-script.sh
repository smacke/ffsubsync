#!/bin/bash

mediapath=$(pwd)
subpath="./subtitles/"
mediafileEXT=".mkv"

for A in $subpath*.srt; do 
  output=${A##$subpath}
  media=$(echo $A |sed -E 's/\.\w\w\.srt//' )
  media=${media##$subpath}
  echo
  echo "-------------------------------------"
  echo "DEBUG: 'subsync $media$mediafileEXT -i $A > $output'"
  subsync $media$mediafileEXT -i $A > $output
  done
