#!/bin/bash

for HW in manila; # lima quito belem santiago bogota;
do
  python richardson_extrapolation.py ${HW}
  echo "${HW} extrapolation finished"
done