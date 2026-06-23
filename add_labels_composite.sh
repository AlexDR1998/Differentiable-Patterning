#!/bin/bash
#
# add_cmy_labels.sh
#
# This script overlays text labels for the composite CMY channels on each segment of the input video.
# The video is assumed to be arranged as 1 row and 3 columns. Each segment contains superimposed CMY channels:
#
#   - In segments 1 and 2: “Cyan”, “Magenta”, and “Yellow” labels are overlaid.
#   - In segment 3: only “Cyan” and “Magenta” labels are overlaid (the Yellow channel is a dummy).
#
# The labels are placed in the top-left corner of each segment with a 10-pixel horizontal padding.
# Vertical positions within each segment are defined with a 30-pixel spacing:
#   - y = 10 for the top label,
#   - y = 40 for the second,
#   - y = 70 for the third (if applicable).
#
# Usage:
#   ./add_cmy_labels.sh input.mp4 output.mp4
#

# Verify that exactly two arguments are provided.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.mp4 output.mp4"
    exit 1
fi

input="$1"
output="$2"

# Set the path to a TrueType font file on your system.
# For example:
#   Ubuntu: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
#   Windows: C:/Windows/Fonts/Arial.ttf  (use forward slashes)
font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Build the FFmpeg filter string using drawtext for each label.
# Note: In FFmpeg expressions,
#   w refers to the full video width.
#   We divide the video width into three segments: each has width w/3.
#
# The formula for the x-coordinate for a segment is:
#   Segment i (0-indexed): x = (i * w/3 + 10)
#
# The y-coordinate for each label is:
#   Cyan:    y = 10
#   Magenta: y = 40
#   Yellow:  y = 70  (skipped for segment 3)
#
filter="\
drawtext=fontfile=${font}:text='SOX2':x=10:y=10:fontsize=22:fontcolor=cyan, \
drawtext=fontfile=${font}:text='TBXT':x=10:y=30:fontsize=22:fontcolor=magenta, \
drawtext=fontfile=${font}:text='SOX17':x=10:y=50:fontsize=22:fontcolor=yellow, \
drawtext=fontfile=${font}:text='Cer1':x='(w/3+10)':y=10:fontsize=22:fontcolor=cyan, \
drawtext=fontfile=${font}:text='Lefty2':x='(w/3+10)':y=30:fontsize=22:fontcolor=magenta, \
drawtext=fontfile=${font}:text='Nodal':x='(w/3+10)':y=50:fontsize=22:fontcolor=yellow, \
drawtext=fontfile=${font}:text='FOXA2':x='(2*w/3+10)':y=10:fontsize=22:fontcolor=cyan, \
drawtext=fontfile=${font}:text='LEF1':x='(2*w/3+10)':y=30:fontsize=22:fontcolor=magenta"

# drawtext=fontfile=${font}:text='Nodal':x='(2*w/3+10)':y=30:fontsize=12:fontcolor=magenta

# Run FFmpeg to overlay the text labels.
ffmpeg -i "$input" -vf "$filter" -c:a copy "$output"
