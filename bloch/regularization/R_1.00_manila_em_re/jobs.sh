#!/bin/bash
RE=Desktop/bloch/regularization/R_1.00_manila_em_re
x=10

for n in 1 2 3 4;
do
  ssh alessandro.tammaro@lab${x} "cd ${RE}; cd e-${n}; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
  echo "Job started on lab${x} for e-${n}"
  x=$(( $x + 1 ))
  sleep 10
done
