#!/bin/sh
# NASA zip first, GitHub mirror if that fails.
set -e
cd "$(dirname "$0")"
mkdir -p data
NASA=https://data.nasa.gov/docs/legacy/CMAPSSData.zip
MIRROR=https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/C-MAPSS-Data

if curl -fsSL "$NASA" -o data/cmapss.zip && unzip -qo data/cmapss.zip -d data; then
    rm data/cmapss.zip
    echo "source: nasa zip"
else
    rm -f data/cmapss.zip
    echo "nasa zip failed, using github mirror"
    for n in 1 2 3 4; do
        for kind in train test RUL; do
            curl -fsSL "$MIRROR/${kind}_FD00$n.txt" -o "data/${kind}_FD00$n.txt"
        done
    done
    echo "source: github mirror"
fi
