#!/bin/bash
for file in replays/*.rep;
do
  ./screp -cmds -header=false $file > JSON/${file/%.rep/.JSON}
done
