#!/bin/bash
#
# add_labels.sh
#
# This script adds static text labels ("Channel 1" to "Channel 8") to an input video.
#
# Usage:
#   ./add_labels.sh input.mp4 output.mp4
#

# Check for proper input arguments.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.mp4 output.mp4"
    exit 1
fi

input="$1"
output="$2"

# Set the font file. Modify this path to point to a valid .ttf font on your system.
font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Build the FFmpeg filter string with multiple drawtext filters.
filter="\
drawtext=fontfile=${font}:text='LMBR':x=10:y=20:fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='Foxa2':x='(w/2+10)':y=20:fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='TBXT':x=10:y='(h/4+20)':fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='Cer1':x='(w/2+10)':y='(h/4+20)':fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='Sox17':x=10:y='(h/2+20)':fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='Lefty2':x='(w/2+10)':y='(h/2+20)':fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='Sox2':x=10:y='(3*h/4+20)':fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='Nodal':x='(w/2+10)':y='(3*h/4+20)':fontsize=24:fontcolor=white
"


# Execute FFmpeg with the built filter.
ffmpeg -i "$input" -vf "$filter" -c:a copy "$output"
