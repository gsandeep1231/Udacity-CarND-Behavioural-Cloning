#!/bin/csh

rm -rf ./tmp
cat myDrivingData/driving_log.csv | awk -F"," '$4 > 0.1 {print ;}' | awk -F"," '{print "rm -rf " $1 ";"}' > ./tmp
cat myDrivingData/driving_log.csv | awk -F"," '$4 > 0.1 {print ;}' | awk -F"," '{print "perl -pi -e '"'"'s#" $1 ".*\\n##g'"'"' myDrivingData/driving_log.csv;"}' >> ./tmp
chmod 755 ./tmp
./tmp
