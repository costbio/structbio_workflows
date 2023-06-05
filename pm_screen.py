# Add a docstring 
from bioservices import UniProt
import io
import pandas as pd
import numpy as np
import subprocess
import os
import glob
import re
import sys
import itertools
import shutil
import pandas as pd
from pathlib import Path
from Bio.PDB import *
from Bio.PDB import PDBList
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from tqdm import tqdm

def getUniprotData(gene_path):
    
    #cwd = os.getcwd()
    #os.chdir(gene_path)
    
    # Read in the txt file containing gene names
    with open(gene_path,"r") as f:
        lines = f.readlines()
        gene_names = [line.rstrip('\n') for line in lines]

    #for filename in os.listdir(gene_path):
    #    genes = os.path.join(gene_path, filename)
    
    #if os.path.isfile(genes):
    #    with open(genes, "r") as file:
    #        gene_names.extend([line.strip() for line in file.readlines()])
        
    if not gene_names:
        raise ValueError(f"No gene names found in directory {gene_path}")    

    # Construct a query string for UniProt search.    
    query = " OR ".join([f"gene_exact:{gene}" for gene in gene_names]) #get genes 

    service = UniProt()

    # Query UniProt and store results in a dataframe.
    df = service.get_df(query, organism="Homo sapiens")
    
    return df

def download_pdb(df, out_path):
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    pdb_rows = []
    data_list = []  # Move this line outside the loop

    for i, row in df.iterrows():
        pdb_value = row['PDB']
        if isinstance(pdb_value, str):
            pdb_list = pdb_value.split(';')
            new_rows = [[row['Entry']]*len(pdb_list),
                        [row['Entry Name']]*len(pdb_list), pdb_list]
            new_df = pd.DataFrame(new_rows).transpose()
            new_df.columns = ["UniProt ID", "Gene Name", "PDB"]
            pdb_rows.append(new_df) 
            pdbl = PDBList()
            for pdb in pdb_list:
                gene = row['Entry'].split(';')[0]
                gene_dir = os.path.join(out_path, gene)
                if not os.path.exists(gene_dir):
                    os.mkdir(gene_dir)   
                pdbl.retrieve_pdb_file(pdb, pdir=gene_dir, file_format='pdb')
                pdb_file = os.path.join(gene_dir,pdb + ".pdb") 
                if not os.path.exists(pdb_file):
                    pdb_file = os.path.join(gene_dir, pdb + ".ent")
                parser = PDBParser(PERMISSIVE=True, QUIET=True)
                for filename in os.listdir(gene_dir):
                    if filename.endswith(".ent"):
                        pdb_id = filename[:-4]  # Remove the ".pdb" extension
                        pdb_file = os.path.join(gene_dir, filename)
                        data = parser.get_structure(pdb_id, pdb_file)
                        header_dict = {"PDB": data.header.get("idcode"),
                                       #"Structure name": data.header.get("name"),
                                       "Resolution": data.header.get("resolution"),
                                       "Has missing residues": data.header.get("has_missing_residues"),
                                       "Structure method": data.header.get("structure_method"),}
                        data_list.append(header_dict)  # Append dictionary to data_list
         
            for dirpath, dirnames, filenames in os.walk(gene_dir):
                for filename in filenames:
                    if filename.endswith(".ent") and filename.startswith("pdb"):
                        old_file_name = os.path.join(dirpath, filename)
                        new_file_name = os.path.join(dirpath, filename[3:])
                        os.rename(old_file_name, new_file_name)
    
    b = pd.concat(pdb_rows)
    res = pd.DataFrame(data_list)  # Use data_list to create res dataframe
    merged_df = pd.merge(b, res, on='PDB', how='outer')
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.dropna()
    merged_df.to_csv(os.path.join(out_path,'Protein_info.csv'))
    return merged_df

def getAlphaFold(merged_df, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    alphafold_ID = 'AF-P28335-F1'
    database_version = 'v2'
    new_values = set(merged_df['UniProt ID'])
    new_IDs = []
    
    # Loop over the unique new values and generate a new ID and folder for each value
    for new_value in new_values:
        # Replace the middle part of the alphafold_ID with the new value
        prefix, middle, suffix = alphafold_ID.split('-')
        middle = new_value
        new_ID = '-'.join([prefix, middle, suffix])
        
        # Create a folder with the name of the middle value inside the output directory
        new_folder = os.path.join(out_path, middle)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        
        # Append the new ID to the list
        new_IDs.append(new_ID)
        
        
    for alphafold_ID in new_IDs:
        # Generate the model and error URLs for the current ID
        model_url = f'https://alphafold.ebi.ac.uk/files/{alphafold_ID}-model_{database_version}.pdb'
        error_url = f'https://alphafold.ebi.ac.uk/files/{alphafold_ID}-predicted_aligned_error_{database_version}.json'

        # Create a folder with the name of the middle value inside the output directory
        middle = alphafold_ID.split('-')[1]
        middle_folder = os.path.join(out_path, middle)
        if not os.path.exists(middle_folder):
            os.mkdir(middle_folder)
            
        # Download the model and error files for the current ID and save them in the appropriate folder
        model_path = os.path.join(middle_folder, f'{alphafold_ID}.pdb')
        error_path = os.path.join(middle_folder, f'{alphafold_ID}.json')
        os.system(f'curl {model_url} -o {model_path}')
        os.system(f'curl {error_url} -o {error_path}')
        
            
        data_list = []
        for protein_name in new_values:
            protein_folder = os.path.join(out_path, protein_name)
            id_to_gene = dict(zip(merged_df['UniProt ID'], merged_df['Gene Name']))
            gene_name = id_to_gene.get(protein_name)
            for filename in os.listdir(protein_folder):
                if filename.endswith(".pdb"):
                    pdb_id = filename[:-4]
                    pdb_file = os.path.join(protein_folder, filename)
                    parser = PDBParser(PERMISSIVE=True, QUIET=True)
                    data = parser.get_structure(pdb_id, pdb_file)
                    header_dict = {"UniProt ID": protein_name,
                                   "Gene Name": gene_name,
                                   "PDB": data.header.get("idcode") or "-",
                                   "Resolution": data.header.get("resolution"),
                                   "Has missing residues": data.header.get("has_missing_residues"),
                                   "Structure method": data.header.get("structure_method")}
                    data_list.append(header_dict)


    a = pd.DataFrame(data_list)
    combined_df = pd.concat([a, merged_df], ignore_index=True)
    combined_df['AlphaFold'] = [False if val != '-' else True for val in combined_df['PDB']]
    combined_df= combined_df.reindex(columns=['UniProt ID', "Gene Name", 'PDB',"AlphaFold","Resolution","Has missing residues","Structure method"])
    return combined_df

def getPockets(out_path, protein_path="/mnt/c/Users/PC/Desktop/Green_Experiment/PDB_download_Green", ds_path="/mnt/c/Users/PC/Desktop/dogsitescorer-2.0.0/", nprocs=4):
    cwd = os.getcwd()
    os.chdir(ds_path)

    pdb_list = []
    for root, dirs, files in os.walk(protein_path):
        for file in files:
            if re.match(r'.+\.(pdb|ent)$', file):
                pdb_list.append(os.path.join(root, file))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    results = {}
    with Pool(processes=nprocs) as pool, tqdm(total=len(pdb_list)) as pbar:
        for model_path in pdb_list:
            try:
                protein_name = os.path.splitext(os.path.basename(model_path))[0]
                protein_folder = os.path.basename(os.path.dirname(model_path))
                output_folder = os.path.join(out_path, protein_folder)
                output_file = os.path.join(output_folder, "{}_{}.pdb".format(protein_folder, protein_name))
                os.makedirs(output_folder, exist_ok=True)
                subprocess.run(["./dogsite", "-p", model_path, "-s", "-i", "-y", "-d", "-w", "4", "-o", output_file])
                results[protein_name] = output_file
                pbar.update(1)
            except Exception as e:
                print("Error processing model:", model_path, ":", str(e))
        
        for dirpath, dirnames, filenames in os.walk(out_path):
            for filename in filenames:
                if "AF" in filename:
                    old_file_name = os.path.join(dirpath, filename)
                    new_file_name = os.path.join(dirpath, "AF" + filename.split("AF")[1])
                    os.rename(old_file_name, new_file_name)
                    
    os.chdir(cwd)
    return results

def parsePockets(folder_path="/mnt/c/Users/PC/Desktop/Green_Experiment/Pockets_DogSiteScore"):
    # Get a list of all subdirectories in the folder
    subdirs = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Define an empty DataFrame
    df = pd.DataFrame()

    # Iterate over the subdirectories and read each text file into a DataFrame
    for subdir in subdirs:
        txt_files = glob.glob(os.path.join(subdir, '*_desc.txt'))
        for txt in txt_files:
            # Read the text file into a DataFrame
            df2 = pd.read_csv(txt, sep='\t')

            # Get the basename of the file
            txt_basename = os.path.basename(txt)
            txt_parts = txt_basename.split('_')
            txt2 = '_'.join(txt_parts[1:])

            # Add columns to the DataFrame with the basename of the file and the folder name
            folder_name = os.path.basename(subdir)
            df2['Protein'] = txt2
            df2['UniProt ID'] = folder_name

            # Concatenate the current DataFrame with the previous ones
            df = pd.concat([df, df2], ignore_index=True)

    # Rename the 'frame' and 'folder' columns and save the DataFrame to a CSV file
    col_c = df.pop('Protein')
    df.insert(0, 'Protein', col_c)
    col_c = df.pop('UniProt ID')
    df.insert(0, 'UniProt ID', col_c)
    df = df.rename(columns={'name': 'Pocket'})
    df['Protein'] = df['Protein'].str[3:]
    df['Protein'] = df['Protein'].str.upper()
    a1=df[["Protein","1"]]= df['Protein'].str.split(".", n=1, expand=True)
    df=df.drop("1", axis=1)
    df['Protein'] = df['Protein'].str.replace('-F1', '')
    gene_to_id = dict(zip(combined_df['UniProt ID'], combined_df['Gene Name']))
    # Add a new column to the DataFrame that contains the corresponding ID for each gene name
    df['Gene Name'] = df['UniProt ID'].map(gene_to_id)
    col = df.pop('Gene Name')
    df.insert(1, 'Gene Name', col)
    df.to_csv(os.path.join(folder_path, 'DrugScore.csv'))

    return df

def getPMsimilarities(combined_df,final_path, out_path= "/mnt/c/Users/PC/Desktop/PocketMatch_v2.1_deneme/",
                      pm_path="/mnt/c/Users/PC/Desktop/PocketMatch_v2.1_deneme/cabbage-file_maker/Sample_pockets/",
                     run_path="/mnt/c/Users/PC/Desktop/PocketMatch_v2.1_deneme/cabbage-file_maker/", 
                      pockets_path="/mnt/c/Users/PC/Desktop/Green_Experiment/Pockets_DogSiteScore/"):
    
    cwd = os.getcwd()
    os.chdir(pm_path)
    pockets = []
    for root, dirs, files in os.walk(pockets_path):
        for file in files:
            if 'res' in file:
                pockets.append(os.path.join(root, file))
    
    for i in pockets:
        subprocess.call("cp %s $PWD" %i ,shell=True)
   
    if not os.path.exists(final_path):
        os.mkdir(final_path)


    pockets_bn = [os.path.basename(pocket) for pocket in pockets]
    
    df_sims_list = list()
    for i in range(len(pockets_bn)):
        for j in range(i+1, len(pockets_bn)):
            df_sim_list = list()
            
    os.chdir(run_path)
    cmd = ' '.join(["bash", "Step0-cabbage.sh", "Sample_pockets/"])
    pocket_match = subprocess.call(cmd,shell=True)
    
    os.chdir(out_path)
    subprocess.call("cp cabbage-file_maker/outfile.cabbage $PWD",shell=True)
    subprocess.run(["./Step3-PM_typeB", "outfile.cabbage", "outfile.cabbage"])
    
    os.chdir(run_path)
    cmd = ' '.join(["bash", "Step0-cabbage.sh", "Sample_pockets/"])
    pocket_match = subprocess.call(cmd,shell=True)
    
    
    os.chdir(out_path)
    subprocess.call("cp cabbage-file_maker/outfile.cabbage $PWD",shell=True)
    subprocess.run(["./Step3-PM_typeB", "outfile.cabbage", "outfile.cabbage"])
    
    for file_path in Path(pm_path).glob('*'):
        try:
            if file_path.is_file():
                file_path.unlink()
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')
    
    
    df = pd.read_table('PocketMatch_score.txt',header=None,delimiter=' ',on_bad_lines='skip') 
    df = df[[0,1,2,3]]
    df.columns = ['Pocket1','Pocket2','Pmin','Pmax']
    df = df.convert_dtypes()
    df= df[df['Pmax'] != 'NULL____']
    df= df[df['Pmin'] != 'NULL____']
    df["Pmin"] = pd.to_numeric(df["Pmin"])
    df["Pmax"] = pd.to_numeric(df["Pmax"])
    df= df[df['Pmax'] != 1]
    df= df[df['Pmin'] != 1]
    df_orig = df
    #df= df[df['Pocket1'].str[:8] != df['Pocket2'].str[:8]]
    #df[["Pocket1","1","2"]]=df['Pocket1'].str.split("___" , expand=True )
    #delete_col=["1","2"]
    #df.drop(delete_col, axis=1)
    #df[["Pocket2","1","2"]]=df['Pocket2'].str.split("___" , expand=True )
    #delete_col=["1","2"]
    #df=df.drop(delete_col, axis=1)
    
    
    dfa=df.sort_values(by='Pmax',ascending=False)
    df1 = dfa[['Pocket1']].copy()
    df1['UniProt ID'] = df1['Pocket1'].apply(lambda x: x.split('_', 1)[0] if (x.startswith('P') or x.startswith('Q')) and '_' in x else '')
    df1['Pocket'] = df1['Pocket1'].apply(lambda x: x.split('_', 1)[1] if (x.startswith('P') or x.startswith('Q')) and '_' in x else x)
    df1 = df1[['UniProt ID', 'Pocket']]
    dfa = dfa.join(df1)
    dfa = dfa.reindex(columns=['UniProt ID', "Pocket","Pocket1","Pocket2","Pmin","Pmax"])
    dfa=dfa.drop("Pocket1", axis=1)
    df2 = dfa[['Pocket2']].copy()
    df2['UniProt ID_2'] = df2['Pocket2'].apply(lambda x: x.split('_', 1)[0] if (x.startswith('P') or x.startswith('Q')) and '_' in x else '')
    df2['Pocket_2'] = df2['Pocket2'].apply(lambda x: x.split('_', 1)[1] if (x.startswith('P') or x.startswith('Q')) and '_' in x else x)
    df2 = df2[['UniProt ID_2', 'Pocket_2']]
    dfa = dfa.join(df2)
    col = dfa.pop('UniProt ID_2')
    dfa.insert(2, 'UniProt ID_2', col)
    col = dfa.pop('Pocket_2')
    dfa.insert(3, 'Pocket_2', col)
    dfa=dfa.drop("Pocket2", axis=1)
    a1=dfa[["Protein","Pocket"]]= dfa['Pocket'].str.split(".", n=1, expand=True)
    col = dfa.pop('Protein')
    dfa.insert(1, 'Protein', col)
    dfa['Pocket'] = dfa['Pocket'].str.replace('pdb_res_', '')
    a1=dfa[["Pocket","1"]]= dfa['Pocket'].str.split(".", n=1, expand=True)
    dfa=dfa.drop("1", axis=1)
    dfa['Protein'] = dfa['Protein'].str.upper()
    a1=dfa[["Protein_2","Pocket_2"]]= dfa['Pocket_2'].str.split(".", n=1, expand=True)
    col = dfa.pop('Protein_2')
    dfa.insert(4, 'Protein_2', col)
    dfa['Pocket_2'] = dfa['Pocket_2'].str.replace('pdb_res_', '')
    a1=dfa[["Pocket_2","1"]]= dfa['Pocket_2'].str.split(".", n=1, expand=True)
    dfa=dfa.drop("1", axis=1)
    dfa['Protein_2'] = dfa['Protein_2'].str.upper()
    dfa['P'] = dfa['Protein'].apply(lambda x: re.findall(r'P\d+', x)[0] if '-' in x and re.findall(r'P\d+', x) else '')
    mask = dfa['UniProt ID'] == ''
    dfa.loc[mask, 'UniProt ID'] = dfa.loc[mask, 'P']
    dfa['P'] = dfa['Protein_2'].apply(lambda x: re.findall(r'P\d+', x)[0] if '-' in x and re.findall(r'P\d+', x) else '')
    mask = dfa['UniProt ID_2'] == ''
    dfa.loc[mask, 'UniProt ID_2'] = dfa.loc[mask, 'P']
    dfa=dfa.drop("P", axis=1)
    #Create a dictionary that maps gene names to IDs
    gene_to_id = dict(zip(combined_df['UniProt ID'], combined_df['Gene Name']))
    # Add a new column to the DataFrame that contains the corresponding ID for each gene name
    dfa['Gene Name'] = dfa['UniProt ID'].map(gene_to_id)
    col = dfa.pop('Gene Name')
    dfa.insert(1, 'Gene Name', col)
    #Create a dictionary that maps gene names to IDs
    gene_to_id = dict(zip(combined_df['UniProt ID'], combined_df['Gene Name']))
    # Add a new column to the DataFrame that contains the corresponding ID for each gene name
    dfa['Gene Name_2'] = dfa['UniProt ID_2'].map(gene_to_id)
    col = dfa.pop('Gene Name_2')
    dfa.insert(5, 'Gene Name_2', col)
    dfa= dfa[dfa['Gene Name']!= dfa['Gene Name_2']]
    df_sims=dfa
    
    df_sims.to_csv(os.path.join(final_path,'df_sims.csv'))
    result=glob.glob(out_path +'/*.txt')
    
    for text_path in result:
        shutil.copy(text_path, final_path)
        os.remove(text_path)
    
    os.chdir(cwd)
    return df_sims

df_uniprot= getUniprotData("/mnt/c/Users/PC/Desktop/GENES/PPI_GreenGene.txt")
df_uniprot.head()

df_uniprot.to_csv('df_uniprot_Green.csv')

df_pdb = download_pdb(df_uniprot,out_path=os.path.join(os.getcwd(),'PDB_download_Green'))

df_pdb_af = getAlphaFold(df_pdb,out_path=os.path.join(os.getcwd(),'PDB_download_Green'))

results = getPockets(out_path=os.path.join(os.getcwd(),'Pockets_DogSiteScore'),nprocs=12)

df_pockets = parsePockets(folder_path=os.path.join(os.getcwd(),'Pockets_DogSiteScore'))

df_sims = getPMsimilarities(df_pdb_af,final_path=os.path.join(os.getcwd(),'PM_GreenExp'),out_path= "/mnt/c/Users/PC/Desktop/PocketMatch_v2.1_deneme/",
                            pm_path="/mnt/c/Users/PC/Desktop/PocketMatch_v2.1_deneme/cabbage-file_maker/Sample_pockets/",
                            run_path="/mnt/c/Users/PC/Desktop/PocketMatch_v2.1_deneme/cabbage-file_maker/", 
                            pockets_path="/mnt/c/Users/PC/Desktop/Green_Experiment/Pockets_DogSiteScore/")

df_sims.to_csv('df_sims_Green.csv')