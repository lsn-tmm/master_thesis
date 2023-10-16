for hw in kolkata auckland;
do
  for R in 0.80 0.90 1.10 1.20;
  do
    cp -r 1.00/R_1.00_qasm_${hw}_raw ${R}/R_${R}_qasm_${hw}_raw
    cd ${R}/R_${R}_qasm_${hw}_raw
    sed -i -e "s/1.00/${R}/g" main.py
    cd ../../
    cp -r 1.00/R_1.00_qasm_${hw}_em ${R}/R_${R}_qasm_${hw}_em
    cd ${R}/R_${R}_qasm_${hw}_em
    sed -i -e "s/1.00/${R}/g" main.py
    cd ../../
    cp -r 1.00/R_1.00_qasm_${hw}_em_re ${R}/R_${R}_qasm_${hw}_em_re
    cd ${R}/R_${R}_qasm_${hw}_em_re
    sed -i -e "s/1.00/${R}/g" main.py
    cd ../../
  done
done
