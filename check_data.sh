#!/bin/sh
# line counts of the official release, stop on any mismatch
cd "$(dirname "$0")/data"
bad=0
while read f want; do
    got=$(wc -l < "$f" | tr -d ' ')
    echo "$f $got (want $want)"
    [ "$got" = "$want" ] || bad=1
done <<LIST
train_FD001.txt 20631
train_FD002.txt 53759
train_FD003.txt 24720
train_FD004.txt 61249
test_FD001.txt 13096
test_FD002.txt 33991
test_FD003.txt 16596
test_FD004.txt 41214
RUL_FD001.txt 100
RUL_FD002.txt 259
RUL_FD003.txt 100
RUL_FD004.txt 248
LIST
exit $bad
