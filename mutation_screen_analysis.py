import argparse
import pandas as pd
import os
import tqdm
from prody import *
import panedr
import numpy as np
import multiprocessing

def analyse_interaction_energies(input_dir):
    # Read the wildtype folder's energies_intEnTotal.csv
    energies_wt = pd.read_csv(os.path.join(input_dir, 'wildtype', 'energies_intEnTotal.csv'))

    # Columns of energies_wt include pairs
    pairs = energies_wt.columns[1:]
    
    # Find out all folders starting with mut in the input directory
    mut_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f)) and f.startswith('mut')]

    # Read the energies_intEnTotal.csv file for each mut folder
    energies_mut = {}
    for folder in tqdm.tqdm(mut_folders):
        energies_mut[folder] = pd.read_csv(os.path.join(input_dir, folder, 'energies_intEnTotal.csv'))

    # For each pair in pairs, construct an array of the wildtype energy and the mutated energies
    energies = {}
    for pair in tqdm.tqdm(pairs):
        energies[pair] = [energies_wt[pair].values[0]]
        for folder in mut_folders:
            if pair in energies_mut[folder]:
                energies[pair].append(energies_mut[folder][pair].values[0])

    # For each energy array, compute a local frustration score, using the following equation:
    # local_frustration = (E_wt-average(E_mut))/(1/len(pairs)*sqrt(sum((E_mut-average(E_mut))^2)))
    local_frustration = {}
    for pair in tqdm.tqdm(pairs):
        if energies[pair][1:]:
            local_frustration[pair] = (energies[pair][0] - sum(energies[pair][1:])/len(energies[pair][1:]))/(1/len(pairs)*sum([(x - sum(energies[pair][1:])/len(energies[pair][1:]))**2 for x in energies[pair][1:]]))**0.5

    # Construct a dataframe out of these scores.
    local_frustration_df = pd.DataFrame(local_frustration, index=['local_frustration']).T

    # Parse system.pdb in the wildtype folder to get the residue names
    system = parsePDB(os.path.join(input_dir, 'wildtype', 'system.pdb'))
    resnames = system.select('protein and name CA').getResnames()

    # Construct a list including <Chain>_<Residue number>_<Residue name> for each residue
    residue_ids = [f'{res.getChid()}_{res.getResnum()}_{res.getResname()}' for res in system.select('protein and name CA')]

    # For each row of local_frustration_df, find the relevant residue_ids for each pair and add them as a new column
    for row in local_frustration_df.index:
        pair = row.split('-')
        local_frustration_df.loc[row, 'pair1'] = residue_ids[int(pair[0])]
        local_frustration_df.loc[row, 'pair2'] = residue_ids[int(pair[1])]
    
    # Save the dataframe to a file in the input directory
    local_frustration_df.to_csv(os.path.join(input_dir, 'local_frustration_mut.csv'))

    # For each pair1 in local_frustration_df, compute the average local frustration score
    # for all pairs that include the same residue
    avg_local_frustration = {}
    for pair1 in residue_ids:
        avg_local_frustration[pair1] = local_frustration_df[(local_frustration_df['pair1'] == pair1)]['local_frustration'].mean()

    # Convert the dictionary to a dataframe
    avg_local_frustration_df = pd.DataFrame(avg_local_frustration, index=['avg_local_frustration']).T

    # Save the dataframe to a file in the input directory
    avg_local_frustration_df.to_csv(os.path.join(input_dir, 'srfi_mut.csv'))

def process_mutation(args):
    folder = args[0]
    input_dir = args[1]
    potential_wt = args[2]
    minim_mut = panedr.edr_to_df(os.path.join(input_dir, folder, 'minim.edr'))
    potential_mut = minim_mut['Potential'].values[-1]
    with open(os.path.join(input_dir, folder, 'mutation.txt'), 'r') as f:
        lines = f.readlines()
        mut = lines[0].strip().split('_')
        res1_index = int(mut[0].split('-')[0])
        res2_index = int(mut[1].split('-')[0])
        res1_aa = mut[0].split('-')[1]
        res2_aa = mut[1].split('-')[1]
        pair_indices = '_'.join(map(str, np.sort(np.asarray([res1_index, res2_index]))))
    return [potential_wt, potential_mut, pair_indices, res1_index, res2_index, res1_aa, res2_aa, folder]

def list_mut_folders(input_dir):
    mut_folders = []
    with os.scandir(input_dir) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.startswith('mut'):
                mut_folders.append(entry.name)
    return mut_folders

def analyse_total_energies(input_dir):
    # Read the wildtype folder's minim.edr
    minim_wt = panedr.edr_to_df(os.path.join(input_dir, 'wildtype', 'minim.edr'))
    potential_wt = minim_wt['Potential'].values[-1]

    # Read the potential energy for each mut folder using multiprocessing
    # Parallelize directory listing using multiprocessing
    with multiprocessing.Pool() as pool:
        mut_folders = pool.map(list_mut_folders, [input_dir])[0]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        potential_mut_list = list(tqdm.tqdm(pool.imap(process_mutation,[[folder,input_dir,potential_wt] for folder in mut_folders]), total=len(mut_folders)))

    # Construct a dataframe out of potential_mut_list
    potential_mut_df = pd.DataFrame(potential_mut_list, columns=['potential_wt','potential', 'pair_indices', 'res1_index', 'res2_index', 'res1_aa', 'res2_aa', 'folder'])

    # Save the dataframe to a file in the input directory
    potential_mut_df.to_csv(os.path.join(input_dir, 'potential_mut.csv'))

    # Compute local frustration scores
    local_frustration = {}
    for pair_indices in tqdm.tqdm(potential_mut_df['pair_indices'].unique()):
        potential_decoys_pair = potential_mut_df[potential_mut_df['pair_indices'] == pair_indices]['potential'].values
        potential_decoys_pair_mean = np.mean(potential_decoys_pair)
        ndecoys_pair = len(potential_mut_df[potential_mut_df['pair_indices'] == pair_indices])
        sum_term = np.sum((potential_decoys_pair - potential_decoys_pair_mean)**2)
        #print(potential_wt, potential_decoys_pair_mean, ndecoys_pair, sum(potential_decoys_pair - potential_decoys_pair_mean))
        local_frustration[pair_indices] = (potential_wt - potential_decoys_pair_mean) / (sum_term/ndecoys_pair)**0.5

    # Construct DataFrame for local frustration scores
    local_frustration_df = pd.DataFrame(local_frustration, index=['frust_index']).T

    # Add residue indices to local frustration DataFrame
    for row in tqdm.tqdm(local_frustration_df.index):
        pair = list(map(int,row.split('_')))
        local_frustration_df.loc[row, 'pair1'] = pair[0]
        local_frustration_df.loc[row, 'pair2'] = pair[1]

    # Save the DataFrame to a file
    local_frustration_df.to_csv(os.path.join(input_dir, 'local_frustration_mut.csv'))

def parse_args():
    parser = argparse.ArgumentParser(description='Mutation screening analysis')
    parser.add_argument('input_dir', type=str, help='Input directory containing the mutation screening results')
    parser.add_argument('--screen_type', type=str, default='pairs', help='Type of mutation screening results')
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir
    screen_type = args.screen_type
    if screen_type == 'pairs':
        analyse_total_energies(input_dir)
    else:
        analyse_interaction_energies(input_dir)

if __name__ == '__main__':
    main()