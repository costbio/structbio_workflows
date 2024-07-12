import argparse
import networkx as nx
import pandas as pd
import os
import tqdm
import numpy as np
from prody import *
from prody import LOGGER
import logging

# Directly modifying logging level for ProDy to prevent printing of noisy debug/warning
# level messages on the terminal.
LOGGER._logger.setLevel(logging.FATAL)

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

    # Configure logging format and file
    loggingFormat = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    logFile = os.path.join(os.path.abspath(outFolder), 'calc.log')

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(loggingFormat, datefmt='%d-%m-%Y:%H:%M:%S')

    # Create console handler and set level to DEBUG if noconsoleHandler is False
    if not noconsoleHandler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Create file handler and set level to DEBUG
    file_handler = logging.FileHandler(logFile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def grinn_pen(grinn_output_folder, out_folder, noconsole_handler=False, seq_sep=None, ie_cutoff=None, ie_cutoff_type=None):
    """
    Construct a protein energy network (PEN) using the gRINN output folder.

    Parameters:
    - grinn_output_folder (str): The folder containing the gRINN output.
    - noconsole_handler (bool): Whether to add a console handler to the logger (default is False).
    - seq_sep (int): The sequence separation for PEN construction (default is None).
    - ie_cutoff (float): The cutoff for interaction energies (default is None).
    - ie_cutoff_type (str): The cutoff type for interaction energies (default is None).
    """

    logger = create_logger(grinn_output_folder, noconsole_handler)

    # Parse system_dry.pdb from gRINN output folder
    pdb_filepath = os.path.join(grinn_output_folder, 'system_dry.pdb')
    if not os.path.exists(pdb_filepath):
        logger.error(f"Could not find {pdb_filepath}")
        return

    system_dry = parsePDB(pdb_filepath)
    res_indices = np.unique(system_dry.getResindices())

    logger.info(f"Number of unique residue indices: {len(res_indices)}")

    # Prepare a list of chain_resnums for each residue in res_indices
    chain_resnums = []
    for res_index in res_indices:
        chain = system_dry.select('resindex %i' % res_index).getChids()[0]
        resnum = system_dry.select('resindex %i' % res_index).getResnums()[0]
        chain_resnums.append(chain+'_'+str(resnum))

    logger.info(f"Number of unique chain_resnums: {len(chain_resnums)}")

    # Check what type of ie_cutoff_type was specified
    if ie_cutoff_type == 'vdw':
        df_ie = pd.read_csv(os.path.join(grinn_output_folder, 'energies_intEnVdW.csv'))
    elif ie_cutoff_type == 'elec':
        df_ie = pd.read_csv(os.path.join(grinn_output_folder, 'energies_intEnElec.csv'))
    else:
        df_ie = pd.read_csv(os.path.join(grinn_output_folder, 'energies_intEnTotal.csv'))

    logger.info(f"Number of unique pairs: {len(df_ie.columns)}")

    # Construct PEN using networkx
    g = nx.Graph()

    logger.info("Constructing PEN...")

    # Add nodes
    for node in chain_resnums:
        g.add_node(node)
    
    logger.info(f"Number of nodes: {len(g.nodes())}")

    # Extract a list of pairs from column names of df_ie
    pairs = []
    for col in df_ie.columns[1:]:
        pairs.append(tuple(col.split('-')))

    # Define a small constant to avoid division by zero
    epsilon = 1e-6

    # Add edges
    for pair in pairs:
        chain_resnum1 = chain_resnums[int(pair[0])]
        chain_resnum2 = chain_resnums[int(pair[1])]
        
        # Get the interaction energy for the current pair
        ie = df_ie[pair[0] + '-' + pair[1]].values[0]
        
        # Only consider pairs that meet the interaction energy cutoff
        if chain_resnum1 != chain_resnum2 and ie <= ie_cutoff:
            # Shift the interaction energy to ensure all values are positive
            shifted_ie = ie + np.abs(np.min(df_ie.values)) + epsilon
            
            # Determine the weight based on shifted interaction energy
            weight = 1 / shifted_ie  # Smaller weight for more favorable interactions
            
            # Add edge to the graph with calculated weight
            g.add_edge(chain_resnum1, chain_resnum2, weight=weight)

    logger.info(f"Number of edges: {len(g.edges())}")

    # Compute degrees of nodes
    degrees = nx.degree(g)
    degrees = dict(degrees)
    df_degrees = pd.DataFrame.from_dict(degrees, orient='index', columns=['degree'])

    # Compute betweenness centrality
    betweenness = nx.betweenness_centrality(g, weight="weight")
    df_betweenness = pd.DataFrame.from_dict(betweenness, orient='index', columns=['betweenness'])

    # Compute closeness centrality
    closeness = nx.closeness_centrality(g)
    df_closeness = pd.DataFrame.from_dict(closeness, orient='index', columns=['closeness'])

    # Compute eigenvector centrality
    eigenvector = nx.eigenvector_centrality(g, weight="weight", max_iter=1000)
    df_eigenvector = pd.DataFrame.from_dict(eigenvector, orient='index', columns=['eigenvector'])

    # If out_folder does not exist, create it
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Save PEN and these centrality measures in out_folder
    df = pd.concat([df_degrees, df_betweenness, df_closeness, df_eigenvector], axis=1, ignore_index=False)
    df.index = df_degrees.index
    df.to_csv(os.path.join(out_folder, 'grinn_pen.csv'), index=True)

    # Save PEN
    nx.write_gml(g, os.path.join(out_folder, 'grinn_pen.gml'))

    logger.info("PEN constructed and saved into out_folder.")


def parse_args():
    parser = argparse.ArgumentParser(description="Construct a protein energy network (PEN), perform basic analyses of it, \
                                     and save results in a folder.")
    parser.add_argument("grinn_output_folder", type=str, help="Folder containing the gRINN output")
    parser.add_argument("outfolder", type=str, help="Output folder")
    parser.add_argument("--noconsole_handler", action="store_true", help="Do not add console handler to the logger")
    parser.add_argument("--seq_sep", type=int, help="Sequence separation for PEN construction")
    parser.add_argument("--ie_cutoff", type=float, help="Cutoff for interaction energies")
    parser.add_argument("--ie_cutoff_type", type=str, help="Cutoff type for interaction energies")
    return parser.parse_args()

def main():
    args = parse_args()

    # Construct PEN
    grinn_pen(
        grinn_output_folder=args.grinn_output_folder,  # Input PDB file
        out_folder=args.outfolder,  # Output folder
        noconsole_handler=args.noconsole_handler,  # Do not add console handler to the logger
        seq_sep=args.seq_sep,  # Sequence separation for PEN construction
        ie_cutoff=args.ie_cutoff,  # Cutoff for interaction energies
        ie_cutoff_type=args.ie_cutoff_type  # Cutoff type for interaction energies
    )

if __name__ == "__main__":
    main()# %%