#!/usr/bin/env bash

rm ../zAggregateData/AllDataLocations/all_data.csv
cat ../zAggregateData/symlinksToDrivingLogs/* >> ../zAggregateData/AllDataLocations/all_data.csv