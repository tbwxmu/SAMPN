## Compare with DeepChem 
In stalll deepchem on Windows from the source code
bash```
git clone https://github.com/deepchem/deepchem.git      # Clone deepchem source code from GitHub
cd deepchem
conda create -n dc python=3.6 pip -you  #creat deepchem envoriment in anaconda
conda install -n dc -y -q -c mikesilva -c deepchem -c rdkit -c conda-forge -c omnia  mdtraj=1.9.1  pdbfixer  joblib=0.11  six=1.11.0  scikit-learn=0.19.1  networkx=2.1  pillow  pandas=0.22.0  nose=1.3.7  nose-timer=0.7.0  flaky=3.3.0  zlib=1.2.11  requests=2.18.4  xgboost=0.6a2  simdna=0.4.2  pbr=3.1.1  setuptools=39.0.1  biopython=1.71 numpy=1.14 rdkit
pip install tensorflow-gpu==1.14
pip install xgboost
set PYTHONPATH=%PYTHONPATH%;C:\deepchem_download_path
```
## Reproducing the table 3 results
```
python lipo_MPNN.py     #this will get the reult of MPN (Deepchem)b in lipohilicity
python delaney_MPNN.py  #this will get the reult of MPN (Deepchem)b in lipohilicity
```
You may get the simmlar but notexact same results as deepchem source code do not deal very well with the random seeds. Even use the same random seeds, MPN(deepchem) may get different results.
