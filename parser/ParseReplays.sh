#!/bin/bash
for file in replays/bot/*.rep;
do
  ./screp -cmds -header=false $file > JSON/${file/%.rep/.JSON}
done
for file in replays/human/*.rep;
do
  ./screp -cmds -header=false $file > JSON/${file/%.rep/.JSON}
done
mv JSON/replays/bot/* JSON/bot/
mv JSON/replays/human/* JSON/human/
rm JSON/bot/\*.JSON
rm JSON/human/\*.JSON
Rscript Converter.R
