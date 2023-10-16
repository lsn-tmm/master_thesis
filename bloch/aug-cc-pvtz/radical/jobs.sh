
#!/bin/bash
RADICAL=Desktop/bloch/aug-cc-pvtz/radical
x=10

for hw in manila lima quito santiago bogota belem;
do
  #ssh alessandro.tammaro@lab${x} "cd ${RADICAL}; cd R_1.00_${hw}_raw; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
  #echo "RAW Job started on lab${x} for ${hw}"
  #x=$(( $x + 1 ))
  #sleep 10
  #ssh alessandro.tammaro@lab${x} "cd ${RADICAL}; cd R_1.00_${hw}_em; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
  #echo "EM Job started on lab${x} for ${hw}"
  #x=$(( $x + 1 ))    
  #sleep 10
  ssh alessandro.tammaro@lab${x} "cd ${RADICAL}; cd R_1.00_${hw}_em_re; source ~/.bashrc; nohup python main.py > main.txt &; echo $! > save_pid.txt;" &
  echo "EM+RE Job started on lab${x} for ${hw}"
  x=$(( $x + 1 ))    
  sleep 10
done
