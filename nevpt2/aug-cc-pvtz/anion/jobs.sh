
#!/bin/bash
ANION=Desktop/freezing_vqe_test/aug-cc-pvtz/anion
x=60

#bash prepare_jobs.sh

for R in 1.00 1.02 1.04 1.16 1.18; #0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98 1.00 1.02 1.04 1.06 1.08 1.10 1.12 1.14 1.16 1.18 1.20;
do
  ssh alessandro.tammaro@lab${x} "cd ${ANION}; cd R_${R}; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
  x=$(( $x + 1 ))
  sleep 5
done
