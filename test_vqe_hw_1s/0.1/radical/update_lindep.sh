LINDEP=0.1

for hw in kolkata auckland qasm_kolkata qasm_auckland;
do
  for R in 1.00
  do
    cd ${R}/R_${R}_${hw}_raw
    sed -i -e "s/0.5/${LINDEP}/g" main.py
    cd ../../
    cd ${R}/R_${R}_${hw}_em
    sed -i -e "s/0.5/${LINDEP}/g" main.py
    cd ../../
    #cd ${R}/R_${R}_${hw}_em_re
    #sed -i -e "s/0.5/${LINDEP}/g" main.py
    #cd ../../
  done
done
