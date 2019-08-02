#MPN
sed -i s/args.attention=True/args.attention=False/g reg_*.py
python reg_log.py
python reg_waterSolLogP.py
python reg_wat.py
sed -i s/1255/3032/g reg_*.py
python reg_log.py
python reg_waterSolLogP.py
python reg_wat.py
sed -i s/3032/11353/g reg_*.py
python reg_log.py
python reg_waterSolLogP.py
python reg_wat.py


#run SAMPN
sed -i s/args.attention=False/args.attention=True/g reg_*.py
python reg_log.py
python reg_waterSolLogP.py
python reg_wat.py
sed -i s/11353/3032/g reg_*.py
python reg_log.py
python reg_waterSolLogP.py
python reg_wat.py
sed -i s/3032/1255/g reg_*.py
python reg_log.py
python reg_waterSolLogP.py
python reg_wat.pyound
