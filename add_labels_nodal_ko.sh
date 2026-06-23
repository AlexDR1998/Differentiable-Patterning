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
drawtext=fontfile=${font}:text='TBXT':x=1:y=1:fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='SOX17':x=1:y=(h/5+1):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='SOX2':x=1:y=(2*h/5+1):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='FOXA2':x=1:y=(3*h/5+1):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='LEF1':x=1:y=(4*h/5+1):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 0h':x=(40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 4h':x=(2*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 8h':x=(4*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 12h':x=(6*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 16h':x=(8*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 20h':x=(10*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 24h':x=(12*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 28h':x=(14*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 32h':x=(16*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 36h':x=(18*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 40h':x=(20*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='KO 44h':x=(22*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white, \
drawtext=fontfile=${font}:text='Baseline':x=(24*w/26+40):y=(h-h/30):fontsize=24:fontcolor=white
"


# Execute FFmpeg with the built filter.
ffmpeg -i "$input" -vf "$filter" -c:a copy "$output"
