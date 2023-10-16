for R in 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98 1.00 1.02 1.04 1.06 1.08 1.10 1.12 1.14 1.16 1.18 1.20;
do
 cd R_${R}
 cp ../../../../nevpt2/6-31g/radical/R_${R}/results.txt ./
 tail -n +2 results_vqe.txt >> results.txt
 echo "${R} finished"
 cd ../
done
