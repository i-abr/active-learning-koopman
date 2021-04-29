#!/bin/zsh

trials=5
Ts=(10 20 30 40 50)
kind=(0 1)
for i in {1..$trials}
do
    for t in $Ts
    do
            for k in $kind
            do
                echo "trial $i horizon $t kind $k"
            done
    done

done
