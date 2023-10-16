
#!/bin/bash
ANION=Desktop/nevpt2_hw/aug-cc-pvqz/anion
x=10

for hw in qasm manila;
do
  for R in 1.10 1.20;
  do
    ssh alessandro.tammaro@lab${x} "cd ${ANION}; cd ${R}; cd R_${R}_${hw}_raw; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
    echo "RAW Job started on lab${x} for ${R}_${hw}"
    x=$(( $x + 1 ))
    sleep 10
    ssh alessandro.tammaro@lab${x} "cd ${ANION}; cd ${R}; cd R_${R}_${hw}_em; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
    echo "EM Job started on lab${x} for ${R}_${hw}"
    x=$(( $x + 1 ))    
    sleep 10
    ssh alessandro.tammaro@lab${x} "cd ${ANION}; cd ${R}; cd R_${R}_${hw}_em_re; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
    echo "EM+RE Job started on lab${x} for ${R}_${hw}"
    x=$(( $x + 1 ))    
    sleep 10
  done
done
