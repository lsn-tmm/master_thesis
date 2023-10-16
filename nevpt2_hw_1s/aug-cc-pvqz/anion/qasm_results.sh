
#!/bin/bash
ANION=Desktop/nevpt2_hw_1s/aug-cc-pvqz/anion
x=10

for hw in kolkata auckland;
do
  for R in 0.80 0.90 1.00 1.10 1.20;
  do
    cd ${R};
    cd R_${R}_qasm_${hw}_raw;
    echo "Results for ${R}_{R}_${hw}_raw ------------------------------"
    tail results.txt
    cd ../../
    cd ${R};
    cd R_${R}_qasm_${hw}_em;
    echo "Results for ${R}_{R}_${hw}_em ------------------------------"
    tail results.txt
    cd ../../
    cd ${R}; 
    cd R_${R}_qasm_${hw}_em_re; 
    echo "Results for ${R}_{R}_${hw}_em_re --------------------------- "
    tail results.txt
    cd ../../
  done
done
