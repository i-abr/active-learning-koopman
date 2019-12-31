#!/bin/zsh

trials=4
Ts=(20 30 40)
kind=(0 1)
for i in {1..$trials}
do
    for t in $Ts
    do
            for k in $kind
            do
                python3 koopman_test.py --T $t --type $k
                echo "trials $i out of $trials"
            done
    done
done
