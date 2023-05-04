import glob
import os
import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Get a list of all .pdb files with their full path in the directory 2021_09_21_charmmGUI_norA_splitPDB
pdb_list = glob.glob('2021_09_21_charmmGUI_norA_splitPDB/*.pdb')

out_folder = '2021_09_21_charmmGUI_norA_splitPDB_dss'

# Create the output folder if it doesn't exist
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# For each .pdb file in the list, call /mnt/d/software/dogsitescorer-2.0.0/dogsite 
def get_pockets_singlecore(pdb_list_chunk):

    for i in tqdm.tqdm(range(0,len(pdb_list_chunk))):
        pdb = pdb_list_chunk[i]

        pdb2 = os.path.basename(pdb)

        p = subprocess.Popen(f'/mnt/d/software/dogsitescorer-2.0.0/dogsite -p {pdb} -o {out_folder}/{pdb2} -i -y 1 -s 1 -w 4 -d 1 > /dev/null', shell=True)
        p.wait()

# Create a pool of nprocs workers.
nprocs = 16

pdb_list2 = [os.path.basename(x) for x in pdb_list]

# Get a list of *_desc.txt files with their full path in the directory 2021_09_21_charmmGUI_norA_splitPDB_dss
txt_list = glob.glob(f'{out_folder}/*_desc.txt')

# Get basenames of all *_desc.txt files
txt_list = [os.path.basename(x) for x in txt_list]

# Replace the _desc.txt with nothing
txt_list_pdbs = [x.replace('_desc.txt', '') for x in txt_list]

# Get a list of pdb files whose filename (without any path info) + "*_desc.txt" are not found in any of the strings in txt_list
pdb_list_names = [x for x in pdb_list2 if x not in txt_list_pdbs]

pdb_list = [os.path.join('2021_09_21_charmmGUI_norA_splitPDB', x) for x in pdb_list_names]

#print(pdb_list[0])
#raise SystemExit(0)

# Split pdb_list into nprocs chunks.
pdb_list_chunks = np.array_split(pdb_list, nprocs)

with ThreadPoolExecutor(max_workers=nprocs) as executor:
    for chunk in pdb_list_chunks:
        future = executor.submit(get_pockets_singlecore, chunk)