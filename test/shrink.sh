#!/bin/bash

#for z in 0.459 0.460
for z in $(LC_ALL=C seq 0.45 0.1 1.5)
do
   echo z: $z
   python /home/marius/Go/phd/structures/structure_bulk322_0K_shrink/experiment.py structure_bulk 10 1 $z
   sleep 1
   rm 1.h5 1.cell 1.json 
   sleep 1
   ln -s /home/marius/Go/phd/structures/structure_bulk322_0K_shrink/_1.h5 /home/marius/Dokumenter/Skole/phd/goofy.git/test/1.h5
   ln -s /home/marius/Go/phd/structures/structure_bulk322_0K_shrink/1.cell /home/marius/Dokumenter/Skole/phd/goofy.git/test/1.cell
   ln -s /home/marius/Go/phd/structures/structure_bulk322_0K_shrink/1.json /home/marius/Dokumenter/Skole/phd/goofy.git/test/1.json
   julia --project=/home/marius/Dokumenter/Skole/phd/goofy.git /home/marius/Dokumenter/Skole/phd/goofy.git/test/predict.jl
done
