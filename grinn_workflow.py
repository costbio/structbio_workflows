# %%
from prody import *
from prody import LOGGER
import numpy as np
import itertools
from itertools import islice
from concurrent import futures
import pyprind
from contextlib import contextmanager
import os, sys, pickle, shutil, pexpect, time, subprocess, panedr, pandas, glob
import logging
from scipy.sparse import lil_matrix
import gromacs
import gromacs.environment
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import mdtraj as md
import pandas as pd
import argparse

# Directly modifying logging level for ProDy to prevent printing of noisy debug/warning
# level messages on the terminal.
LOGGER._logger.setLevel(logging.FATAL)

# %%
def create_logger(outFolder, noconsoleHandler=False):
    """
    Create a logger with specified configuration.

    Parameters:
    - outFolder (str): The folder where log files will be saved.
    - noconsoleHandler (bool): Whether to add a console handler to the logger (default is False).

    Returns:
    - logger (logging.Logger): The configured logger object.
    """
    # If the folder does not exist, create it
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
    # Configure logging format
    loggingFormat = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    logFile = os.path.join(os.path.abspath(outFolder), 'calc.log')
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(loggingFormat, datefmt='%d-%m-%Y:%H:%M:%S')

    # Create console handler and set level to DEBUG
    if not noconsoleHandler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(logFile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# %%
def run_gromacs_simulation(pdb_filepath, mdp_files_folder, out_folder, ff_folder, nofixpdb, solvate, npt, lig, lig_gro_file, lig_itp_file, logger, nt=1):
    """
    Run a GROMACS simulation workflow.

    Parameters:
    - pdb_filepath (str): The path to the input PDB file.
    - mdp_files_folder (str): The folder containing the MDP files.
    - out_folder (str): The folder where output files will be saved.
    - nofixpdb (bool): Whether to fix the PDB file using pdbfixer (default is True).
    - logger (logging.Logger): The logger object for logging messages.
    - nt (int): Number of threads for GROMACS commands (default is 1).
    - ff_folder (str): The folder containing the force field files (default is None).

    Returns:
    - None
    """

    gromacs.environment.flags['capture_output'] = "file"
    gromacs.environment.flags['capture_output_filename'] = os.path.join(out_folder, "gromacs.log")

    logger.info(f"Running GROMACS simulation for PDB file: {pdb_filepath}")

    if nofixpdb:
        fixed_pdb_filepath = pdb_filepath
    else:
    # Fix PDB file
        fixer = PDBFixer(filename=pdb_filepath)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        pdb_filename = os.path.basename(pdb_filepath)
        fixed_pdb_filepath = os.path.join(out_folder, "protein.pdb")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdb_filepath, 'w'))
        logger.info("PDB file fixed.")
        system = parsePDB(fixed_pdb_filepath)
        writePDB(fixed_pdb_filepath, system.select('protein'))

    if ff_folder is not None:
        ff = ff_folder
    else:
        ff = "amber99sb-ildn"

    # Run GROMACS commands
    try:
        gromacs.pdb2gmx(f=fixed_pdb_filepath, o=os.path.join(out_folder, "protein.pdb"), 
                        p=os.path.join(out_folder, "topol.top"), i=os.path.join(out_folder,"posre.itp"),
                          ff=ff, water="tip3p", heavyh=True, ignh=True)
        logger.info("pdb2gmx command completed.")
        next_pdb = "protein.pdb"

        if lig:
            logger.info("Running ligand mode...")
            lig_itp_outfolder_path = os.path.join(out_folder, "ligand.itp")
            lig_itp_outfolder_path = os.path.abspath(lig_itp_outfolder_path)
            lig_gro_outfolder_path = os.path.join(out_folder, "ligand.gro")
            lig_gro_outfolder_path = os.path.abspath(lig_gro_outfolder_path)
            lig_pdb_outfolder_path = os.path.join(out_folder, "ligand.pdb")
            lig_pdb_outfolder_path = os.path.abspath(lig_pdb_outfolder_path)
            shutil.copy(lig_gro_file, lig_gro_outfolder_path)
            shutil.copy(lig_itp_file, lig_itp_outfolder_path)
            logger.info("Ligand gro and itp files copied.")
            # Convert gro of ligand to pdb
            gromacs.editconf(f=lig_gro_outfolder_path, o=lig_pdb_outfolder_path)
            logger.info("Ligand gro file converted to pdb.")
            # Create protein-ligand complex
            protein = parsePDB(os.path.join(out_folder, "protein.pdb"))
            ligand = parsePDB(os.path.join(out_folder, "ligand.pdb"))
            lig_chids = ligand.getChids()
            lig_code = ligand.getResnames()[0]
            # Set "Z" as chain ID for ligand as a chain id is required in later stages of the workflow.
            ligand.setChids(['Z']*len(lig_chids))
            complex = protein + ligand
            writePDB(os.path.join(out_folder, "complex.pdb"), complex)
            logger.info("Protein-ligand complex created.")

            # Supplement topology file with ligand topology
            f = open(os.path.join(out_folder, "topol.top"), "r")
            topol_lines = f.readlines()
            f.close()
            write_flag = False
            stop_write = False
            new_lines = list()
            for line in topol_lines:
                new_lines.append(line)
                if line.startswith("#include"):
                    write_flag = True
                if write_flag and not stop_write:
                    new_lines.append(f"#include \"{lig_itp_outfolder_path}\"\n")
                    write_flag = False
                    stop_write = True

            new_lines.append(f"{lig_code}     1\n")

            f = open(os.path.join(out_folder, "topol.top"), "w")
            f.writelines(new_lines)
            f.close()
            logger.info("Supplemented topology file with ligand topology.")

            next_pdb = "complex.pdb"

            # Create the name of the group for the protein+ligand complex for the index file.
            index_group_select = f'"Protein" | "{lig_code}"'
            index_group_name = f"Protein_{lig_code}"
        
        gromacs.make_ndx(f=os.path.join(out_folder, next_pdb), o=os.path.join(out_folder, "index.ndx"), input=(index_group_select,'q'))
        
        shutil.copy(os.path.join(out_folder, "topol.top"), os.path.join(out_folder, "topol_dry.top"))
        logger.info("Topology file copied.")
        #source_folder = os.getcwd()
        #files = os.listdir(source_folder)

        # Filter files starting with "posre" and ending with ".itp"
        #posre_files = [file for file in files if file.startswith("posre") and file.endswith(".itp")]
        # Move each posre file to the out_folder
        #for file in posre_files:
        #    source_file_path = os.path.join(source_folder, file)
        #    dest_file_path = os.path.join(out_folder, file)
        #    shutil.move(source_file_path, dest_file_path)
        
        if solvate:
            gromacs.editconf(f=os.path.join(out_folder, next_pdb), n=os.path.join(out_folder, "index.ndx"), 
                             o=os.path.join(out_folder, "boxed.pdb"), bt="cubic", c=True, d=1.0, princ=True, input=('0','0','0'))
            logger.info("editconf command completed.")
            gromacs.solvate(cp=os.path.join(out_folder, "boxed.pdb"), cs="spc216", p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "solvated.pdb"))
            logger.info("solvate command completed.")
            gromacs.grompp(f=os.path.join(mdp_files_folder, "ions.mdp"), c=os.path.join(out_folder, "solvated.pdb"), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "ions.tpr"))
            logger.info("grompp for ions command completed.")
            gromacs.genion(s=os.path.join(out_folder, "ions.tpr"), o=os.path.join(out_folder, "solvated_ions.pdb"), p=os.path.join(out_folder, "topol.top"), neutral=True, conc=0.15, input=('SOL','q'))
            logger.info("genion command completed.")
            next_pdb = "solvated_ions.pdb"
        else:
            gromacs.editconf(f=os.path.join(out_folder, next_pdb), n=os.path.join(out_folder, 'index.ndx'), 
                             o=os.path.join(out_folder, "boxed.pdb"), bt="cubic", c=True, box=[999,999,999], princ=True, input=(index_group_name, index_group_name, index_group_name))
            logger.info("editconf command completed.")
            next_pdb = "boxed.pdb"
        
        if next_pdb == "solvated_ions.pdb":
            gromacs.grompp(f=os.path.join(mdp_files_folder, "minim.mdp"), c=os.path.join(out_folder, next_pdb), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "minim.tpr"))
        if next_pdb == "boxed.pdb":
            gromacs.grompp(f=os.path.join(mdp_files_folder, "minim_vac.mdp"), c=os.path.join(out_folder, next_pdb), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "minim.tpr"))

        logger.info("grompp for minimization command completed.")
        gromacs.mdrun(deffnm="minim", v=True, c=os.path.join(out_folder, "minim.pdb"), s=os.path.join(out_folder,"minim.tpr"), 
                      e=os.path.join(out_folder,"minim.edr"), g=os.path.join(out_folder,"minim.log"), 
                      o=os.path.join(out_folder,"minim.trr"), x=os.path.join(out_folder,"minim.xtc"), nt=nt)
        logger.info("mdrun for minimization command completed.")
        gromacs.trjconv(f=os.path.join(out_folder, 'minim.pdb'),o=os.path.join(out_folder, 'minim.pdb'), s=os.path.join(out_folder, next_pdb), input=('0','q'))
        logger.info("trjconv for minimization command completed.")
        next_pdb = "minim.pdb"
        gromacs.trjconv(f=os.path.join(out_folder,next_pdb),o=os.path.join(out_folder, "traj.xtc"))

        if npt:
            gromacs.grompp(f=os.path.join(mdp_files_folder, "npt.mdp"), c=os.path.join(out_folder, next_pdb), 
                           r=os.path.join(out_folder, next_pdb), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "npt.tpr"), maxwarn=10)
            logger.info("grompp for NPT command completed.")
            gromacs.mdrun(deffnm="npt", v=True, c=os.path.join(out_folder, "npt.pdb"), s=os.path.join(out_folder,"npt.tpr"), nt=nt, pin='on', 
            x=os.path.join(out_folder, "npt.xtc"), e=os.path.join(out_folder, "npt.edr"), o=os.path.join(out_folder, "npt.trr"))
            logger.info("mdrun for NPT command completed.")
            gromacs.trjconv(f=os.path.join(out_folder, 'npt.pdb'), o=os.path.join(out_folder, 'npt.pdb'), s=os.path.join(out_folder, 'solvated_ions.pdb'), input=('0','q'))
            logger.info("trjconv for NPT command completed.")
            gromacs.trjconv(s=os.path.join(out_folder, 'npt.tpr'), f=os.path.join(out_folder, 'npt.xtc'), o=os.path.join(out_folder, 'traj.xtc'), input=(index_group_name,))
            logger.info("trjconv for NPT to XTC conversion command completed.")
            next_pdb = "npt.pdb"

        gromacs.trjconv(f=os.path.join(out_folder, next_pdb), o=os.path.join(out_folder, 'system_dry.pdb'), s=os.path.join(out_folder, next_pdb), n=os.path.join(out_folder, 'index.ndx'), input=(index_group_name,))
        logger.info(f"trjconv for {next_pdb} to DRY PDB conversion command completed.")
        
        gromacs.trjconv(f=os.path.join(out_folder, 'traj.xtc'), o=os.path.join(out_folder, 'traj_dry.xtc'), s=os.path.join(out_folder, 'system_dry.pdb'), 
                        n=os.path.join(out_folder, 'index.ndx'), input=(index_group_name,))
        logger.info(f"trjconv for traj.xtc to traj_dry.xtc conversion command completed.")

        # Convert npt.xtc to npt.dcd
        traj = md.load(os.path.join(out_folder, 'traj_dry.xtc'), top=os.path.join(out_folder, "system_dry.pdb"))
        traj.save_dcd(os.path.join(out_folder, 'traj_dry.dcd'))

        logger.info("GROMACS simulation completed successfully.")
    except Exception as e:
        logger.error(f"Error encountered during GROMACS simulation: {str(e)}")

# %%
# A method for suppressing terminal output temporarily.
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# %%
def filterInitialPairsSingleCore(args):
    outFolder = args[0]
    pairs = args[1]
    initPairFilterCutoff = args[2]

    with suppress_stdout():
        system = parsePDB(os.path.join(outFolder, "system_dry.pdb"))

    # Define a method for initial filtering of a single pair.
    def filterInitialPair(pair):
        com1 = calcCenter(system.select("resindex %i" % pair[0]))
        com2 = calcCenter(system.select("resindex %i" % pair[1]))
        dist = calcDistance(com1, com2)
        if dist <= initPairFilterCutoff:
            return pair
        else:
            return None

    # Get a list of included pairs after initial filtering.
    filterList = []
    progbar = pyprind.ProgBar(len(pairs))
    for pair in pairs:
        filtered = filterInitialPair(pair)
        if filtered is not None:
            filterList.append(pair)
        progbar.update()

    return filterList

# %%
def perform_initial_filtering(outFolder, source_sel, target_sel, initPairFilterCutoff, numCores, logger):
    """
    Perform initial filtering of residue pairs based on distance.

    Parameters:
    - outFolder (str): The folder where output files will be saved.
    - initPairFilterCutoff (float): The distance cutoff for initial filtering.
    - numCores (int): The number of CPU cores to use for parallel processing.
    - logger (logging.Logger): The logger object for logging messages.

    Returns:
    - initialFilter (list): A list of residue pairs after initial filtering.
    """
    logger.info("Performing initial filtering...")

    # Get the path to the PDB file (system.pdb) from outFolder
    pdb_file = os.path.join(outFolder, "system_dry.pdb")

    # Parse PDB file
    system = parsePDB(pdb_file)
    numResidues = system.numResidues()
    source = system.select(source_sel)
    target = system.select(target_sel)

    sourceResids = np.unique(source.getResindices())
    numSource = len(sourceResids)

    targetResids = np.unique(target.getResindices())
    numTarget = len(targetResids)

    # Generate all possible unique pairwise residue-residue combinations
    pairProduct = itertools.product(sourceResids, targetResids)
    pairSet = set()
    for x, y in pairProduct:
        if x != y:
            pairSet.add(frozenset((x, y)))

    # Prepare a pairSet list
    pairSet = [list(pair) for pair in list(pairSet)]

    # Get a list of pairs within a certain distance from each other, based on the initial structure.
    initialFilter = []

    # Split the pair set list into chunks according to number of cores
    # Reduce numCores if necessary.
    if len(pairSet) < numCores:
        numCores = len(pairSet)
    
    pairChunks = np.array_split(list(pairSet), numCores)

    # Start a concurrent futures pool, and perform initial filtering.
    with futures.ProcessPoolExecutor(numCores) as pool:
        try:
            initialFilter = pool.map(filterInitialPairsSingleCore, [[outFolder, pairChunks[i], initPairFilterCutoff] for i in range(0, numCores)])
            initialFilter = list(initialFilter)
            
            # initialFilter may contain empty lists, remove them.
            initialFilter = [sublist for sublist in initialFilter if sublist]

            # Flatten the list of lists
            if len(initialFilter) > 1:
                initialFilter = np.vstack(initialFilter)
        finally:
            pool.shutdown()

    initialFilter = list(initialFilter)
    initialFilter = [pair for pair in initialFilter if pair is not None]
    logger.info('Initial filtering... Done.')
    logger.info('Number of interaction pairs selected after initial filtering step: %i' % len(initialFilter))

    initialFilterPickle = os.path.join(os.path.abspath(outFolder), "initialFilter.pkl")
    with open(initialFilterPickle, 'wb') as f:
        pickle.dump(initialFilter, f)

    return initialFilter

# %%
# A method to get a string containing chain or seg ID, residue name and residue number
# given a ProDy parsed PDB Atom Group and the residue index
def getChainResnameResnum(pdb,resIndex):
	# Get a string for chain+resid+resnum when supplied the residue index.
	selection = pdb.select('resindex %i' % resIndex)
	chain = selection.getChids()[0]
	chain = chain.strip(' ')
	segid = selection.getSegnames()[0]
	segid = segid.strip(' ')

	resName = selection.getResnames()[0]
	resNum = selection.getResnums()[0]
	if chain:
		string = ''.join([chain,str(resName),str(resNum)])
	elif segid:
		string = ''.join([segid,str(resName),str(resNum)])
	return [chain,segid,resName,resNum,string]

# %%
def calculate_interaction_energies(outFolder, initialFilter, numCoresIE, logger):
    """
    Calculate interaction energies for residue pairs.

    Parameters:
    - outFolder (str): The folder where output files will be saved.
    - numCoresIE (int): The number of CPU cores to use for interaction energy calculation.
    - logger (logging.Logger): The logger object for logging messages.

    Returns:
    - edrFiles (list): List of paths to the EDR files generated during calculation.
    """
    logger.info("Calculating interaction energies...")

    # Read necessary files from outFolder
    pdb_file = os.path.join(outFolder, 'system_dry.pdb')
    top_file = os.path.join(outFolder, 'topol_dry.top')
    xtc_file = os.path.join(outFolder, 'traj_dry.xtc')

    # Modify atom serial numbers to account for possible PDB files with more than 99999 atoms
    system = parsePDB(pdb_file)
    system.setSerials(np.arange(1, system.numAtoms() + 1))

    system_dry = system.select('protein or nucleic or lipid or hetero and not water and not resname SOL and not ion')
    system_dry = system_dry.select('not resname SOL')

    indicesFiltered = np.unique(np.hstack(initialFilter))
    allSerials = {}

    for index in indicesFiltered:
        residue = system_dry.select('resindex %i' % index)
        lenSerials = len(residue.getSerials())
        if lenSerials > 14:
            residueSerials = residue.getSerials()
            allSerials[index] = [residueSerials[i:i + 14] for i in range(0, lenSerials, 14)]
        else:
            allSerials[index] = np.asarray([residue.getSerials()])

    # Write a standard .ndx file for GMX
    filename = os.path.join(outFolder, 'interact.ndx')
    gromacs.make_ndx(f=os.path.join(outFolder, 'system_dry.pdb'), o=filename, input=('q',))

    # Append our residue groups to this standard file!
    with open(filename, 'a') as f:
        for key in allSerials:
            f.write('[ res%i ]\n' % key)
            if type(allSerials[key][0]).__name__ == 'ndarray':
                for line in allSerials[key][0:]:
                    f.write(' '.join(list(map(str, line))) + '\n')
            else:
                f.write(' '.join(list(map(str, allSerials))) + '\n')

    # Write the .mdp files necessary for GMX
    mdpFiles = []

    # Divide pairsFiltered into chunks so that each chunk does not contain
    # more than 200 unique residue indices.
    pairsFilteredChunks = []
    if len(np.unique(np.hstack(initialFilter))) <= 60:
        pairsFilteredChunks.append(initialFilter)
    else:
        i = 2
        maxNumRes = len(np.unique(np.hstack(initialFilter)))
        while maxNumRes >= 60:
            pairsFilteredChunks = np.array_split(initialFilter, i)
            chunkNumResList = [len(np.unique(np.hstack(chunk))) for chunk in pairsFilteredChunks]
            maxNumRes = np.max(chunkNumResList)
            i += 1

    for pair in initialFilter:
        if pair not in np.vstack(pairsFilteredChunks):
            logger.exception('Missing at least one residue in filtered residue pairs. Please contact the developer.')
        
    i = 0
    for chunk in pairsFilteredChunks:
        filename = str(outFolder)+'/interact'+str(i)+'.mdp'
        f = open(filename,'w')
        #f.write('cutoff-scheme = group\n')
        f.write('cutoff-scheme = Verlet\n')
        #f.write('epsilon-r = %f\n' % soluteDielectric)

        chunkResidues = np.unique(np.hstack(chunk))

        resString = ''
        for res in chunkResidues:
            resString += 'res'+str(res)+' '

        #resString += ' SOL'

        f.write('energygrps = '+resString+'\n')

        # Add energygroup exclusions.
        #energygrpExclString = 'energygrp-excl ='

        # GOTTA COMMENT OUT THE FOLLOWING DUE TO TOO LONG LINE ERROR IN GROMPP
        # for key in allSerials:
        # 	energygrpExclString += ' res%i res%i' % (key,key)

        #energygrpExclString += ' SOL SOL'
        #f.write(energygrpExclString)

        f.close()
        mdpFiles.append(filename)
        i += 1

    # Call gromacs pre-processor (grompp) and make a new TPR file for each pair
    edrFiles = []
    for i, chunk in enumerate(pairsFilteredChunks):
        mdpFile = os.path.join(outFolder, f'interact{i}.mdp')
        tprFile = mdpFile.rstrip('.mdp') + '.tpr'
        edrFile = mdpFile.rstrip('.mdp') + '.edr'

        gromacs.grompp(f=mdpFile, n=os.path.join(outFolder, 'interact.ndx'), p=top_file, c=pdb_file, o=tprFile, maxwarn=20)

        gromacs.mdrun(v=True, s=tprFile, c=pdb_file, e=edrFile, nt=numCoresIE, pin='on', rerun=xtc_file)

        edrFiles.append(edrFile)

        subprocess.Popen(['rm', 'md.log']).wait()

        logger.info('Completed calculation percentage: ' + str((i + 1) / len(mdpFiles) * 100))

    return edrFiles, pairsFilteredChunks

# %%
def parse_interaction_energies(edrFiles, pairsFilteredChunks, outFolder, logger):
    """
    Parse interaction energies from EDR files and save the results.

    Parameters:
    - edrFiles (list): List of paths to the EDR files.
    - outFolder (str): The folder where output files will be saved.
    - logger (logging.Logger): The logger object for logging messages.
    """

    system = parsePDB(os.path.join(outFolder, 'system_dry.pdb'))
    
    logger.info('Parsing GMX energy output... This may take a while...')
    df = panedr.edr_to_df(os.path.join(outFolder, 'interact0.edr'))
    logger.info('Parsed 1 EDR file.')

    for i in range(1, len(edrFiles)):
        edrFile = edrFiles[i]
        df_pair = panedr.edr_to_df(edrFile)

        # Remove already parsed columns
        df_pair_columns = df_pair.columns
        df_pair = df_pair[list(set(df_pair_columns) - set(df.columns))]

        df = pd.concat([df, df_pair], axis=1)
        logger.info('Parsed %i out of %i EDR files...' % (i + 1, len(edrFiles)))

    logger.info('Collecting energy results...')
    energiesDict = dict()

    for i in range(len(pairsFilteredChunks)):
        pairsFilteredChunk = pairsFilteredChunks[i]
        energiesDictChunk = dict()

        for pair in pairsFilteredChunk:
            #res1_string = getChainResnameResnum(system, pair[0])[-1]
            #res2_string = getChainResnameResnum(system, pair[1])[-1]
            energyDict = dict()

            # Lennard-Jones Short Range interaction
            column_stringLJSR1 = 'LJ-SR:res%i-res%i' % (pair[0], pair[1])
            column_stringLJSR2 = 'LJ-SR:res%i-res%i' % (pair[1], pair[0])
            if column_stringLJSR1 in df.columns:
                column_stringLJSR = column_stringLJSR1
            elif column_stringLJSR2 in df.columns:
                column_stringLJSR = column_stringLJSR2
            else:
                logger.exception('At least one required residue interaction was not found in the pair interaction '
                                 'energy output. Please contact the developer.')
                raise SystemExit(0)

            # Lennard-Jones 1-4 interaction
            column_stringLJ141 = 'LJ-14:res%i-res%i' % (pair[0], pair[1])
            column_stringLJ142 = 'LJ-14:res%i-res%i' % (pair[1], pair[0])
            if column_stringLJ141 in df.columns:
                column_stringLJ14 = column_stringLJ141
            elif column_stringLJ142 in df.columns:
                column_stringLJ14 = column_stringLJ142
            else:
                logger.exception('At least one required residue interaction was not found in the pair interaction '
                                 'energy output. Please contact the developer.')
                raise SystemExit(0)

            # Coulombic Short Range interaction
            column_stringCoulSR1 = 'Coul-SR:res%i-res%i' % (pair[0], pair[1])
            column_stringCoulSR2 = 'Coul-SR:res%i-res%i' % (pair[1], pair[0])
            if column_stringCoulSR1 in df.columns:
                column_stringCoulSR = column_stringCoulSR1
            elif column_stringCoulSR2 in df.columns:
                column_stringCoulSR = column_stringCoulSR2
            else:
                logger.exception('At least one required residue interaction was not found in the pair interaction '
                                 'energy output. Please contact the developer.')
                raise SystemExit(0)

            # Coulombic Short Range interaction
            column_stringCoul141 = 'Coul-14:res%i-res%i' % (pair[0], pair[1])
            column_stringCoul142 = 'Coul-14:res%i-res%i' % (pair[1], pair[0])
            if column_stringCoul141 in df.columns:
                column_stringCoul14 = column_stringCoul141
            elif column_stringCoul142 in df.columns:
                column_stringCoul14 = column_stringCoul142
            else:
                logger.exception('At least one required residue interaction was not found in the pair interaction '
                                 'energy output. Please contact the developer.')
                raise SystemExit(0)

            # Convert energy units from kJ/mol to kcal/mol
            kj2kcal = 0.239005736
            enLJSR = np.asarray(df[column_stringLJSR].values) * kj2kcal
            enLJ14 = np.asarray(df[column_stringLJ14].values) * kj2kcal
            enLJ = [enLJSR[j] + enLJ14[j] for j in range(len(enLJSR))]
            energyDict['VdW'] = enLJ

            enCoulSR = np.asarray(df[column_stringCoulSR].values) * kj2kcal
            enCoul14 = np.asarray(df[column_stringCoul14].values) * kj2kcal
            enCoul = [enCoulSR[j] + enCoul14[j] for j in range(len(enCoulSR))]
            energyDict['Elec'] = enCoul

            energyDict['Total'] = [energyDict['VdW'][j] + energyDict['Elec'][j] for j in range(len(energyDict['VdW']))]

            #key1 = res1_string + '-' + res2_string
            #key1 = key1.replace(' ', '')
            #key2 = res2_string + '-' + res1_string
            #key2 = key2.replace(' ', '')
            #energiesDictChunk[key1] = energyDict
            #energiesDictChunk[key2] = energyDict

            # Also use residue indices - may come handy later on for some analyses
            key1_alt = str(pair[0]) + '-' + str(pair[1])
            energiesDictChunk[key1_alt] = energyDict

        energiesDict.update(energiesDictChunk)
        logger.info('Collected %i out of %i results' % (i + 1, len(pairsFilteredChunks)))

    logger.info('Collecting results...')

    # Prepare data tables from parsed energies and save to files
    df_total = pd.DataFrame()
    df_elec = pd.DataFrame()
    df_vdw = pd.DataFrame()

    for key, value in energiesDict.items():
        df_total[key] = value['Total']
        df_elec[key] = value['Elec']
        df_vdw[key] = value['VdW']

    logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnTotal.csv'))
    df_total.to_csv(os.path.join(outFolder, 'energies_intEnTotal.csv'))
    logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnElec.csv'))
    df_elec.to_csv(os.path.join(outFolder, 'energies_intEnElec.csv'))
    logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnVdW.csv'))
    df_vdw.to_csv(os.path.join(outFolder, 'energies_intEnVdW.csv'))

    logger.info('Pickling results...')

    # Split the dictionary into chunks for pickling
    def chunks(data, SIZE=10000):
        it = iter(data)
        for i in range(0, len(data), SIZE):
            yield {k: data[k] for k in islice(it, SIZE)}

    enDicts = list(chunks(energiesDict, 1000))

    intEnPicklePaths = []

    # Pickle the chunks
    for i in range(len(enDicts)):
        fpath = os.path.join(outFolder, 'energies_%i.pickle' % i)
        with open(fpath, 'wb') as file:
            logger.info('Pickling to energies_%i.pickle...' % i)
            pickle.dump(enDicts[i], file)
            intEnPicklePaths.append(fpath)

    logger.info('Pickling results... Done.')

def cleanUp(outFolder, logger):
    """
    Clean up the output folder by removing unnecessary files.

    Parameters:
    - outFolder (str): The folder where output files will be saved.
    """
    # Cleaning up the output folder
    logger.info('Cleaning up...')

    # Delete all NAMD-generated energies file from output folder
    for item in glob.glob(os.path.join(outFolder, '*_energies.log')):
        os.remove(item)

    for item in glob.glob(os.path.join(outFolder, '*temp*')):
        os.remove(item)

    # Delete all GROMACS-generated energies file from output folder
    for item in glob.glob(os.path.join(outFolder, 'interact*')):
        os.remove(item)

    for item in glob.glob(os.path.join(outFolder, '*.trr')):
        os.remove(item)

    if os.path.exists(os.path.join(outFolder, 'traj.dcd')):
        os.remove(os.path.join(outFolder, 'traj.dcd'))

    for item in glob.glob(os.path.join(os.getcwd(), '#*#')):
        os.remove(item)

    for item in glob.glob(os.path.join(outFolder, '#*#')):
        os.remove(item)

    logger.info('Cleaning up... completed.')
                      
def source_gmxrc(gmxrc_path):
    """
    Sources the GMXRC script to set up GROMACS environment variables.
    
    Args:
    - gmxrc_path (str): Path to the GMXRC script.
    
    Returns:
    - dict: Dictionary containing the environment variables set by GMXRC.
    """
    # Run the command to execute the GMXRC script
    subprocess.call(gmxrc_path,shell=True)
    
    # Retrieve the environment variables set by GMXRC
    gmx_env_vars = {}
    for line in subprocess.check_output("env", shell=True).splitlines():
        line = line.decode("utf-8")
        key, _, value = line.partition("=")
        gmx_env_vars[key] = value
    
    return gmx_env_vars

def run_grinn_workflow(pdb_file, mdp_files_folder, out_folder, ff_folder, init_pair_filter_cutoff, nofixpdb=False, top=False, toppar=False, nointeraction=False, solvate=False, npt=False, 
                       source_sel="all", target_sel="all", lig=False, lig_gro_file=None, lig_itp_file=None, nt=1, gmxrc_path='/usr/local/gromacs/bin/GMXRC',
                       noconsole_handler=False):

    start_time = time.time()  # Start the timer
    gmx_env_vars = source_gmxrc(gmxrc_path)
    #print("GROMACS_DIR:", gmx_env_vars.get("GROMACS_DIR"))

    # If source_sel is None, set it to an appropriate selection
    if source_sel is None:
        source_sel = "not water and not resname SOL and not ion"

    # If target_sel is None, set it to an appropriate selection
    if target_sel is None:
        target_sel = "not water and not resname SOL and not ion"

    if type(source_sel) == list:
        if len(source_sel) > 1:
            source_sel = ' '.join(source_sel)
        else:
            source_sel = source_sel[0]

    if type(target_sel) == list:
        if len(target_sel) > 1:
            target_sel = ' '.join(target_sel)
        else:
            target_sel = target_sel[0]

    logger = create_logger(out_folder, noconsole_handler)
    logger.info('### gRINN workflow started ###')
    # Print the command-line used to call this workflow to the log file
    logger.info('gRINN workflow was called as follows: ')
    logger.info(' '.join(sys.argv))

    # Check whether a topology file as well as toppar folder is provided
    if top and toppar:
        logger.info('Topology file and toppar folder provided. Using provided topology file and toppar folder.')
        logger.info('Copying topology file to output folder...')
        shutil.copy(top, os.path.join(out_folder, 'topol_dry.top'))
        logger.info('Copying toppar folder to output folder...')
        shutil.copytree(toppar, os.path.join(out_folder, 'toppar'))
        logger.info('Copying input pdb_file to output_folder as "system.pdb"...')
        shutil.copy(pdb_file, os.path.join(out_folder, 'system_dry.pdb'))
        logger.info('Generating traj.xtc file from input pdb_file...')
        gromacs.trjconv(f=os.path.join(out_folder, 'system_dry.pdb'), o=os.path.join(out_folder, 'traj_dry.xtc'))

    else:
        run_gromacs_simulation(pdb_file, mdp_files_folder, out_folder, ff_folder, nofixpdb, solvate, npt, lig, lig_gro_file, lig_itp_file, logger, nt)
    
    if nointeraction:
        logger.info('Not calculating interaction energies as per user request.')

    else:
        initialFilter = perform_initial_filtering(out_folder, source_sel, target_sel, init_pair_filter_cutoff, 4, logger)
        edrFiles, pairsFilteredChunks = calculate_interaction_energies(out_folder, initialFilter, nt, logger)
        parse_interaction_energies(edrFiles, pairsFilteredChunks, out_folder, logger)

    cleanUp(out_folder, logger)
    elapsed_time = time.time() - start_time  # Calculate the elapsed time    
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    logger.info('Elapsed time: {:.2f} seconds'.format(elapsed_time))
    logger.info('### gRINN workflow completed successfully ###')
    # Clear handlers to avoid memory leak
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Run gRINN workflow")
    parser.add_argument("pdb_file", type=str, help="Input PDB file")
    parser.add_argument("mdp_files_folder", type=str, help="Folder containing the MDP files")
    parser.add_argument("out_folder", type=str, help="Output folder")
    parser.add_argument("--nofixpdb", action="store_true", help="Fix PDB file using pdbfixer")
    parser.add_argument("--initpairfiltercutoff", type=float, default=10, help="Initial pair filter cutoff (default is 10)")
    parser.add_argument("--nointeraction", action="store_true", help="Do not calculate interaction energies")
    parser.add_argument("--solvate", action="store_true", help="Run solvation")
    parser.add_argument("--npt", action="store_true", help="Run NPT equilibration")
    parser.add_argument("--source_sel", nargs="+", type=str, help="Source selection")
    parser.add_argument("--target_sel", nargs="+", type=str, help="Target selection")
    parser.add_argument("--gmxrc_path", type=str, help="Path to the GMXRC script")
    parser.add_argument("--nt", type=int, default=1, help="Number of threads for GROMACS commands (default is 1)")
    parser.add_argument("--noconsole_handler", action="store_true", help="Do not add console handler to the logger")
    parser.add_argument("--ff_folder", type=str, help="Folder containing the force field files")
    parser.add_argument('--top', type=str, help='Topology file')
    parser.add_argument('--toppar', type=str, help='Toppar folder')
    parser.add_argument('--lig', action='store_true', help='Ligand mode')
    parser.add_argument('--lig_gro_file', type=str, help='Ligand gro file')
    parser.add_argument('--lig_itp_file', type=str, help='Ligand itp file')
    return parser.parse_args()

def main():
    args = parse_args()
    run_grinn_workflow(args.pdb_file, args.mdp_files_folder, args.out_folder, args.ff_folder, args.initpairfiltercutoff, args.nofixpdb, args.top, args.toppar, args.nointeraction, args.solvate, args.npt, 
                       args.source_sel, args.target_sel, args.lig, args.lig_gro_file, args.lig_itp_file, args.nt, args.gmxrc_path, args.noconsole_handler)

if __name__ == "__main__":
    main()# %%
