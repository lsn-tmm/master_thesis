for hw in belem bogota lima quito santiago;
do
 #cp -r R_1.00_manila_raw R_1.00_${hw}_raw
 #cd R_1.00_${hw}_raw
 #sed -i -e "s/manila/${hw}/g" main.py
 #cd .. 
 #cp -r R_1.00_manila_em R_1.00_${hw}_em
 #cd R_1.00_${hw}_em
 #sed -i -e "s/manila/${hw}/g" main.py
 #cd ..
 cp -r R_1.00_manila_em_re R_1.00_${hw}_em_re
 cd R_1.00_${hw}_em_re
 sed -i -e "s/manila/${hw}/g" main.py
 cd ..
done
