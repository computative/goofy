#!/bin/bash

for filename in e2e.jl unit.jl
do
	echo $filename
	julia --project=/home/marius/Dokumenter/Skole/phd/goofy.git/ /home/marius/Dokumenter/Skole/phd/goofy.git/test/$filename
done

