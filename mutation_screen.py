from prody import *
from grinn_workflow import *
import tqdm, os, sys, argparse, logging, time, subprocess
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait

def three_to_one(aa):
    if aa == 'ALA':
        return 'A'
    elif aa == 'ARG':
        return 'R'
    elif aa == 'ASN':
        return 'N'
    elif aa == 'ASP':
        return 'D'
    elif aa == 'CYS':
        return 'C'
    elif aa == 'GLN':
        return 'Q'
    elif aa == 'GLU':
        return 'E'
    elif aa == 'GLY':
        return 'G'
    elif aa == 'HIS':
        return 'H'
    elif aa == 'ILE':
        return 'I'
    elif aa == 'LEU':
        return 'L'
    elif aa == 'LYS':
        return 'K'
    elif aa == 'MET':
        return 'M'
    elif aa == 'PHE':
        return 'F'
    elif aa == 'PRO':
        return 'P'
    elif aa == 'SER':
        return 'S'
    elif aa == 'THR':
        return 'T'
    elif aa == 'TRP':
        return 'W'
    elif aa == 'TYR':
        return 'Y'
    elif aa == 'VAL':
        return 'V'
    else:
        return 'X'
    
one_letter_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def call_grinn_workflow_single_core(args):

    mutations = args[0]
    pdb_file = args[1]
    output_dir = args[2]
    init_pair_filter_cutoff = args[3]
    nointeraction = args[4]
    faspr_path = args[5]
    mdp_files_folder = args[6]
    gmxrc_path = args[7]
    main_logger = args[8]
    
    # Loop over each mutation to generate mutated sequences
    for (i, mutation_string, seq) in tqdm.tqdm(mutations):

        main_logger.info('Running grinn_workflow for mutation {}'.format(mutation_string))
        # Create a new directory for the mutated structure
        os.makedirs(os.path.join(output_dir, f'mut_{i}'), exist_ok=True)

        main_logger.info('Generating mutated structure for mutation {}'.format(mutation_string))
        # Write the mutated sequence to a file
        with open(os.path.join(output_dir, f'mut_{i}', 'sequence.txt'), 'w') as f:
            f.write(seq)

        main_logger.info('Running FASPR for mutation {}'.format(mutation_string))
        # Run FASPR to generate mutated structure
        # Usage: ./FASPR -i input.pdb -o output.pdb [-s sequence.txt] to load a sequence file
        subprocess.run([faspr_path, '-i', pdb_file, '-o', os.path.join(output_dir, f'mut_{i}', 'mutated.pdb'), '-s', os.path.join(output_dir, f'mut_{i}', 'sequence.txt')],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        main_logger.info('FASPR completed for mutation {}'.format(mutation_string))
        # Run grinn_workflow for the mutated structure
        run_grinn_workflow(os.path.join(output_dir,f'mut_{i}','mutated.pdb'), mdp_files_folder, os.path.join(output_dir, f'mut_{i}'),
                           init_pair_filter_cutoff=init_pair_filter_cutoff, nointeraction=nointeraction, source_sel="all", target_sel="all", nt=1, gmxrc_path=gmxrc_path, noconsole_handler=True)
        
        # Write the mutated sequence to a file
        with open(os.path.join(output_dir, f'mut_{i}', 'sequence.txt'), 'w') as f:
            f.write(seq)

        # Write the mutation_string to a file
        with open(os.path.join(output_dir, f'mut_{i}', 'mutation.txt'), 'w') as f:
            f.write(mutation_string)
        
        main_logger.info('grinn_workflow completed for mutation {}'.format(mutation_string))

def systematic_mutation_screen(pdb_file, output_dir, nointeraction, mdp_files_folder, faspr_path, one_letter_list, gmxrc_path=None, nt=1):

    # Parse the wild-type structure
    pdb = parsePDB(pdb_file)

    # Create a list of all the residues in the structure
    resnames = pdb.select('protein and name CA').getResnames()

    # Create a list of residue indices
    resindices = pdb.select('protein and name CA').getResindices()

    # Create a list to store mutations to be made
    mutations = []

    # Iterate over each residue in the structure
    for i, resname in enumerate(resnames):

        # Get a list of amino acids other than aa
        aa_one = three_to_one(resname)

        # Mutation candidates
        mutation_candidates = [aa for aa in one_letter_list if aa != aa_one]

        # Randomly select a subset of mutation candidates of five amino acids
        mutation_candidates = np.random.choice(mutation_candidates, 5, replace=False)
        
        # Iterate over each amino acid
        for aa in mutation_candidates:
            # Get wild-type sequence
            wt_seq = ''.join([three_to_one(resname) for resname in resnames])
        
            # Create a copy of the wild-type sequence
            seq = list(wt_seq)
        
            # Replace the wild-type amino acid with the mutated amino acid
            seq[i] = aa
        
            # Join the list to create the mutated sequence
            seq = ''.join(seq)
            mutations.append(i, str(resindices[i]+'-'+aa, seq))

    #TODO - Run FASPR and grinn_workflow for each mutation

def pairs_mutation_screen(pdb_file, output_dir, init_pair_filter_cutoff, nointeraction, ndecoys, 
                          seq_sep, mdp_files_folder, faspr_path, one_letter_list, main_logger, gmxrc_path=None, nt=1):
    
    main_logger.info('### pairs_mutation_screen workflow started ###')
    # Parse the wild-type structure
    pdb = parsePDB(pdb_file)

    main_logger.info('Running pairs_mutation_screen')
    # Find which residues are in contact with each other from wildtype's energies_intEnTotal.csv
    energies_wt = pd.read_csv(os.path.join(output_dir, 'wildtype', 'energies_intEnTotal.csv'))
    pairs = energies_wt.columns[1:]

    main_logger.info('Number of pairs: {}'.format(len(pairs)))
    # Remove pairs for which seq_sep is not satisfied
    pairs = [pair for pair in pairs if abs(int(pair.split('-')[0])-int(pair.split('-')[1])) >= seq_sep]

    main_logger.info('Number of pairs after removing pairs with seq_sep < {}: {}'.format(seq_sep, len(pairs)))
    # Create a list to store mutations to be made
    mutations = []

    # Create a list to store pairs to be mutated
    mutation_pairs = []

    # Create a list of all the residues in the structure
    resnames = pdb.select('protein and name CA').getResnames()
    
    main_logger.info('Generating decoy mutations')
    i = 0
    # For each pair, generate a list of decoy mutations
    for pair in tqdm.tqdm(pairs):

        res1, res2 = pair.split('-')
        res1 = int(res1)
        res2 = int(res2)
        aa1 = three_to_one(pdb.select('protein and name CA').getResnames()[res1])
        aa2 = three_to_one(pdb.select('protein and name CA').getResnames()[res2])

        # Randomly select a residue from one_letter_list that is not the wild-type amino acid aa1
        mutation_candidates1 = [aa for aa in one_letter_list if aa != aa1]

        # Randomly select a residue from one_letter_list that is not the wild-type amino acid aa2
        mutation_candidates2 = [aa for aa in one_letter_list if aa != aa2]

        main_logger.info('Generating decoy mutations for pair {}'.format(pair))
        j = 0
        while j < ndecoys:
            # Create a copy of the wild-type sequence
            seq = list(''.join([three_to_one(resname) for resname in resnames]))

            # Randomly select one from mutation_candidates1 and one from mutation_candidates2
            aa1_mutated = np.random.choice(mutation_candidates1)
            aa2_mutated = np.random.choice(mutation_candidates2)
            seq[int(res1)] = aa1_mutated
            seq[int(res2)] = aa2_mutated
            seq = ''.join(seq)

            mutation_pair = [(res1, aa1_mutated), (res2, aa2_mutated)]
            mutation_pair_string = str(res1)+'-'+aa1_mutated+'_'+str(res2)+'-'+aa2_mutated
            
            if mutation_pair in mutation_pairs:
                continue
            else:
                j = j + 1
            
            mutations.append((i,mutation_pair_string,seq))

            i = i + 1
            main_logger.info('Generated decoy mutation {}'.format(mutation_pair_string))

    main_logger.info('Generated {} decoy mutations'.format(i))
    # Divide mutations into nt chunks
    mutation_chunks = np.array_split(np.asarray(mutations), nt)

    main_logger.info('Running grinn_workflow for each mutation')
    # Start a concurrent futures pool, and perform initial filtering.
    with ProcessPoolExecutor(max_workers=nt) as pool:
        futures = []
        for i in range(0, nt):
            future = pool.submit(call_grinn_workflow_single_core, [mutation_chunks[i], pdb_file, output_dir, init_pair_filter_cutoff,
                                                                    nointeraction, faspr_path, mdp_files_folder,gmxrc_path, main_logger])
            futures.append(future)
        
        wait(futures)

    main_logger.info('### pairs_mutation_screen workflow completed successfully ###')
            
def random_mutation_screen(pdb_file, output_dir, nointeraction, mdp_files_folder, faspr_path, one_letter_list, gmxrc_path=None, nt=1):
    # Parse the wild-type structure
    pdb = parsePDB(pdb_file)

    # Create a list of all the residues in the structure
    resnames = pdb.select('protein and name CA').getResnames()

    # Create a list to store mutations to be made
    mutations = []

    # Generate 100 random sequences
    for i in tqdm.tqdm(range(100)):

        # Create a copy of the wild-type sequence
        seq = list(''.join([three_to_one(resname) for resname in resnames]))

        # Generate a random protein sequence
        for j in range(len(seq)):
            seq[j] = np.random.choice(one_letter_list)

        # Join the list to create the mutated sequence
        seq = ''.join(seq)

        # Append the mutated sequence to the list of mutations
        mutations.append((i,'',seq))

    #TODO - Run FASPR and grinn_workflow for each mutation

def mutation_screen(pdb_file, output_dir, screen_type, init_pair_filter_cutoff, ndecoys, nointeraction, seq_sep, 
                    mdp_files_folder, faspr_path, one_letter_list, gmxrc_path=None, nt=1):

    if screen_type == 'systematic':
        # Run grinn_workflow for the wild-type structure
        run_grinn_workflow(pdb_file, mdp_files_folder, os.path.join(output_dir, 'wildtype'), init_pair_filter_cutoff, 
                       nointeraction=nointeraction, source_sel="all", target_sel="all", nt=nt, gmxrc_path=gmxrc_path, noconsole_handler=True)
        systematic_mutation_screen(pdb_file, output_dir, nointeraction, mdp_files_folder, faspr_path, one_letter_list, gmxrc_path, nt)

    elif screen_type == 'random':
        # Run grinn_workflow for the wild-type structure
        run_grinn_workflow(pdb_file, mdp_files_folder, os.path.join(output_dir, 'wildtype'), init_pair_filter_cutoff,  
                       nointeraction=nointeraction, source_sel="all", target_sel="all", nt=nt, gmxrc_path=gmxrc_path, noconsole_handler=True)
        random_mutation_screen(pdb_file, output_dir, nointeraction, mdp_files_folder, faspr_path, one_letter_list, gmxrc_path, nt)

    elif screen_type == 'pairs':
        # Run grinn_workflow for the wild-type structure
        # Special case, interactions computed for wild-type structure
        main_logger = create_logger(os.path.join(output_dir))
        start_time = time.time()  # Start the timer
        main_logger.info('### mutation_screen workflow started ###')
        main_logger.info("### The following arguments were used: ###")
        main_logger.info(' '.join(sys.argv))
        main_logger.info('#'*50)
        main_logger.info('Running grinn_workflow for the wild-type structure')

        ### FOR TESTING
        # Run input wild-type through FASPR to see its effect on the output

        # Parse the wild-type structure
        pdb = parsePDB(pdb_file)

        # Create a list of all the residues in the structure
        resnames = pdb.select('protein and name CA').getResnames()

        # If the folder does not exist, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(os.path.join(output_dir, 'wildtype')):
            os.makedirs(os.path.join(output_dir, 'wildtype'))
            
        # Write the wild-type sequence to a file
        with open(os.path.join(output_dir, 'wildtype', 'sequence.txt'), 'w') as f:
            f.write(''.join([three_to_one(resname) for resname in resnames]))

        # Usage: ./FASPR -i input.pdb -o output.pdb [-s sequence.txt] to load a sequence file
        subprocess.run([faspr_path, '-i', pdb_file, '-o', os.path.join(output_dir, 'wildtype', 'wildtype.pdb'), '-s', os.path.join(output_dir, 'wildtype', 'sequence.txt')],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        run_grinn_workflow(os.path.join(output_dir, 'wildtype', 'wildtype.pdb'), mdp_files_folder, os.path.join(output_dir, 'wildtype'), init_pair_filter_cutoff, 
                           nointeraction=False, source_sel="all", target_sel="all", nt=nt, gmxrc_path=gmxrc_path)
        
        main_logger.info('Running pairs_mutation_screen')

        pairs_mutation_screen(pdb_file, output_dir, init_pair_filter_cutoff, nointeraction, ndecoys, 
                              seq_sep, mdp_files_folder, faspr_path, one_letter_list, main_logger, gmxrc_path, nt)

        main_logger.info('### mutation_screen workflow completed successfully ###')
        elapsed_time = time.time() - start_time  # Calculate the elapsed time    
        print("Elapsed time: {:.2f} seconds".format(elapsed_time))
        main_logger.info('Elapsed time: {:.2f} seconds'.format(elapsed_time))
        main_logger.info('### mutation_screen workflow completed successfully ###')
        # Clear handlers to avoid memory leak
        for handler in main_logger.handlers:
            handler.close()
            main_logger.removeHandler(handler)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Mutation screening')
    parser.add_argument('pdb_file', type=str, help='Input PDB file')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('screen_type', type=str, choices=['systematic', 'random','pairs'], help='Type of mutation screen')
    parser.add_argument('--initpairfiltercutoff', type=float, default=10, help='Initial pair cutoff for pairs mutation screen')
    parser.add_argument('--ndecoys', type=int, default=100, help='Number of decoys for random mutation screen')
    parser.add_argument('--nointeraction', action='store_true', help='Do not include interaction energies in the analysis')
    parser.add_argument('--seq_sep', type=int, default=5, help='Sequence separation for pairs mutation screen')
    parser.add_argument('--mdp_files_folder', type=str, default='mdp_files', help='Folder containing mdp files')
    parser.add_argument('--faspr_path', type=str, default='/mnt/d/repos/FASPR/FASPR', help='Path to FASPR executable')
    parser.add_argument("--gmxrc_path", type=str, help="Path to the GMXRC script")
    parser.add_argument("--nt", type=int, default=1, help="Number of threads for GROMACS commands (default is 1)")
    return parser.parse_args()

def main():
    args = parse_args()
    mutation_screen(args.pdb_file, args.output_dir, args.screen_type, args.initpairfiltercutoff, args.ndecoys, args.nointeraction, 
                    args.seq_sep, args.mdp_files_folder, args.faspr_path, one_letter_list, args.gmxrc_path, args.nt)

if __name__ == '__main__':
    main()