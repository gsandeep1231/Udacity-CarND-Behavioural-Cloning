#!/bin/csh
perl -pi -e 's/\/User.*IMG/IMG/g' data/driving_log.csv 
find  ./data/IMG/ -name "center*"  | wc
wc data/driving_log.csv 
