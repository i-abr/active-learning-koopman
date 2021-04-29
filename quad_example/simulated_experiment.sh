#!/bin/zsh

trials=10
Ts=(20 25 30 35 40 45 50 55 60)
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
