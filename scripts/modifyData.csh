#!/bin/csh
perl -pi -e 's/\/User.*IMG/IMG/g' data/driving_log.csv 
find  ./data/IMG/ -name "center*"  | wc
wc data/driving_log.csv 
#cat driving_log.csv | awk -F"," '{ if ($4>-0.33) {print ;} }' > driving_log.csv.new
#cat driving_log.csv | awk -F"," '{ if ($4<-0.33) {print $1",,,"$4*0.6","$5","$6","$7} }' >> driving_log.csv.new 

