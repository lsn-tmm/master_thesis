for hw in manila qasm;
do
  for R in 0.90 1.10 1.20;
  do
    cp -r 1.00/R_1.00_${hw}_raw ${R}/R_${R}_${hw}_raw
    cd ${R}/R_${R}_${hw}_raw
    sed -i -e "s/1.00/${R}/g" main.py
    cd ../../
    cp -r 1.00/R_1.00_${hw}_em ${R}/R_${R}_${hw}_em
    cd ${R}/R_${R}_${hw}_em
    sed -i -e "s/1.00/${R}/g" main.py
    cd ../../
    cp -r 1.00/R_1.00_${hw}_em_re ${R}/R_${R}_${hw}_em_re
    cd ${R}/R_${R}_${hw}_em_re
    sed -i -e "s/1.00/${R}/g" main.py
    cd ../../
  done
done
