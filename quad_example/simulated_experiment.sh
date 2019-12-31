#!/bin/zsh

trials=10
for i in {1..$trials}
do
    python3 al_koopman_test.py
    echo "trials $i out of $trials"
done
