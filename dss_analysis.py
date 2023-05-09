# %%
import pandas
import tqdm
import os
import glob
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import seaborn as sns
import biotite.structure.io.pdb as pdb
import scipy.cluster.hierarchy as sch

# %%
# Get a list of all *_desc.txt files with their full path in the directory 2021_09_21_charmmGUI_norA_splitPDB_dss
txt_list = glob.glob('2021_09_21_charmmGUI_norA_splitPDB_dss/*_desc.txt')

# Parse each of these files into a single dataframe
df = pandas.DataFrame()
for i in tqdm.tqdm(range(0,len(txt_list))):
    txt = txt_list[i]
    df2 = pandas.read_csv(txt, sep='\t')

    # Get the basename of the file
    txt2 = os.path.basename(txt)

    # Add a column to the dataframe with the basename of the file
    df2['frame'] = txt2
    df = pandas.concat([df,df2], ignore_index=True)

# %%
# Extract the number in frame column and strore it instead in frame column.
df['frame'] = df['frame'].str.extract('(\d+)').astype(int)

# %%
df.to_csv('df_pockets_2021_09_21_charmmGUI_norA.csv', index=False)

# %%
# Read in df_pockets_2021_09_21_charmmGUI_norA.csv
df_pockets = pandas.read_csv('df_pockets_2021_09_21_charmmGUI_norA.csv')

# %%
# Remove rows with drugScore values lower than 0.5
df_pockets = df_pockets[df_pockets['drugScore'] > 0.5]

# %%
# Summarize df_pockets
df_pockets.describe()

# %%
def annot_pocket_res(df_pockets):
    # Iterate over all rows in df_pockets
    for index, row in tqdm.tqdm(df_pockets.iterrows(), total=df_pockets.shape[0]):
        # Get the name and frame
        name = row['name']
        frame = row['frame']

        pocket_pdb = f'frame_{frame}.pdb_res_{name}.pdb'
        
        # Load the pocket PDB file
        pocket = pdb.PDBFile.read(os.path.join('2021_09_21_charmmGUI_norA_splitPDB_dss',pocket_pdb))

        # Find out unique residue numbers
        res_nums = list(map(int,[pocket.get_structure()[0][i].res_id for i in range(0,len(pocket.get_structure()[0]))]))

        # Get unique values in res_nums
        unique_res_nums = np.unique(res_nums)

        unique_res_str = ' '.join(unique_res_nums.astype(str))

        # Add unique_res_str to df_pockets as a new column
        df_pockets.loc[index, 'pocket_res'] = unique_res_str

    return df_pockets

# %%
ncores = 20

# Split df_pockets into ncores chunks.
df_pockets_chunks = np.array_split(df_pockets, ncores)

# Call annot_pocket_res() on each chunk using multiprocessing
from multiprocessing import Pool
with Pool(ncores) as p:
    df_pockets = pandas.concat(p.map(annot_pocket_res, df_pockets_chunks))

# Save df_pockets to a CSV file
# ds_05: drugScore > 0.5
# wpr: with pocket residues
df_pockets.to_csv('df_pockets_ds_05_wpr.csv', index=True)


# %%
# (Optional) Remove rows that contain more than a single underscore in the name column
df_pockets = df_pockets[~df_pockets['name'].str.contains('_.*_')]

# %%
master_res_list = list()
# Iterate over all rows in df_pockets
for index, row in tqdm.tqdm(df_pockets.iterrows(), total=df_pockets.shape[0]):
    # Convert pocket_res to a list containing integers
    pocket_res = list(map(int, row['pocket_res'].split(' ')))

    # Add pocket_res to master_res_list
    master_res_list.extend(pocket_res)

# %%
# Get how frequently each value is observed in master_res_list in a data frame
df_res_freq = pandas.DataFrame(pandas.Series(master_res_list).value_counts())

# %%
df_res_freq

# %%
# Read in df_pockets_ds_05_wpr.csv
df_pockets_ds_05_wpr = pandas.read_csv('df_pockets_ds_05_wpr.csv')

# %%
# (Optional) Remove rows that contain more than a single underscore in the name column (to exclude subpockets)
#df_pockets_ds_05_wpr = df_pockets_ds_05_wpr[~df_pockets_ds_05_wpr['name'].str.contains('_.*_')]

# %%
# (Optional) Keep only rows that contain at least two underscores in the name columns (to only keep subpockets)
df_pockets_ds_05_wpr = df_pockets_ds_05_wpr[df_pockets_ds_05_wpr['name'].str.contains('_.*_.*')]

# %%
# Summarize df_pockets
df_pockets_ds_05_wpr.head()

# %%
# Get a list of all values in pocket_res column
pocket_res_list = df_pockets_ds_05_wpr['pocket_res'].tolist()

# Convert each element in this list to a list
pocket_res_list = [x.split(' ') for x in pocket_res_list]

# Stack all lists into a single list
pocket_res_list = [item for sublist in pocket_res_list for item in sublist]

# Convert this list to a list of integers
pocket_res_list = [int(x) for x in pocket_res_list]

# Get a list of all unique values in pocket_res_list
pocket_res_list_unique = list(set(pocket_res_list))

# %%
# Create a dataframe that will have a column for each unique value in pocket_res_list_unique, and another column for frame
df_pockets_ds_05_wpr_pocket_res = pandas.DataFrame(columns=['frame'] + pocket_res_list_unique)

# For each row in df_pockets_ds_05_wpr, create a new row in df_pockets_ds_05_wpr_pocket_res, and add a 1 to the column that corresponds to the pocket_res value
for index, row in tqdm.tqdm(df_pockets_ds_05_wpr.iterrows(), total=df_pockets_ds_05_wpr.shape[0]):
    # Get the pocket_res column value
    pocket_res = row['pocket_res']
    # Convert this value to a list
    pocket_res = pocket_res.split(' ')
    # Convert each element in this list to an integer
    pocket_res = [int(x) for x in pocket_res]
    # Create a new row in df_pockets_ds_05_wpr_pocket_res, and add a 1 to the column that corresponds to the pocket_res value
    df_pockets_ds_05_wpr_pocket_res.loc[index] = [row['frame']] + [1 if x in pocket_res else 0 for x in pocket_res_list_unique]

# Remove columns that include only zeros
df_pockets_ds_05_wpr_pocket_res = df_pockets_ds_05_wpr_pocket_res.loc[:, (df_pockets_ds_05_wpr_pocket_res != 0).any(axis=0)]


# %%
# Remove columns that include zeros for more than 90% of the rows
df_pockets_ds_05_wpr_pocket_res = df_pockets_ds_05_wpr_pocket_res.loc[:, (df_pockets_ds_05_wpr_pocket_res == 0).mean() < 0.9]

# Remove rows that include zeros for more than 90% of the columns
# Calculate the percentage of zeros in each row
zeros_percent = (df_pockets_ds_05_wpr_pocket_res == 0).sum(axis=1) / df_pockets_ds_05_wpr_pocket_res.shape[1]

# Filter out rows with more than 90% of zeros
df_pockets_ds_05_wpr_pocket_res = df_pockets_ds_05_wpr_pocket_res[zeros_percent < 0.9]

# %%
# Cluster the rows in df_pockets_ds_05_wpr_pocket_res based on all columns except the first column using seaborn.clustermap
g = sns.clustermap(df_pockets_ds_05_wpr_pocket_res.iloc[:,1:], metric='euclidean', method='ward', cmap='Blues', figsize=(20,20))

# retrieve clusters using fcluster 
#d = sch.distance.pdist(df_pockets_ds_05_wpr_pocket_res.iloc[:,1:])
#L = sch.linkage(d, method='ward')
L = g.dendrogram_row.linkage
# 0.2 can be modified to retrieve more stringent or relaxed clusters
clusters = sch.fcluster(L, 200, 'distance')

print('number of clusters: ', len(np.unique(clusters)))
# clusters indices correspond to incides of original df

df_index = []
cluster_no = []

for i,cluster in enumerate(clusters):
    print(df_pockets_ds_05_wpr_pocket_res.index[i], cluster)
    df_index.append(df_pockets_ds_05_wpr_pocket_res.index[i])
    cluster_no.append(cluster)

df_index_clusters = pandas.DataFrame(list(zip(df_index,clusters)),columns=['Index','cluster_no'])

# %% 
# For each cluster, get the number of rows in df_pockets_ds_05_wpr_pocket_res that belong to this cluster
for cluster in np.unique(clusters):
    df_index_cluster = df_index_clusters[df_index_clusters['cluster_no'] == cluster]
    index_cluster = df_index_cluster['Index'].tolist()

    # Get the rows in df_pockets_ds_05_wpr that belong to this cluster
    df_pockets_ds_05_wpr_cluster = df_pockets_ds_05_wpr[df_pockets_ds_05_wpr.index.isin(index_cluster)]

    # Get the rows in df_pockets_ds_05_wpr_pocket_res that belong to this cluster
    #df_pockets_ds_05_wpr_pocket_res_cluster = df_pockets_ds_05_wpr_pocket_res[df_pockets_ds_05_wpr_pocket_res.index.isin(index_cluster)]

    # Sort the rows in df_pockets_ds_05_wpr_cluster by descending value of Druggability Score
    # These may be used as representatives of the cluster, and potential pockets for drug design
    df_pockets_ds_05_wpr_cluster = df_pockets_ds_05_wpr_cluster.sort_values(by='drugScore',ascending=False)
    print('cluster no: ', cluster)
    print('frame: ', df_pockets_ds_05_wpr_cluster['frame'].tolist()[0])
    print('pocket_res: ', df_pockets_ds_05_wpr_cluster['pocket_res'].tolist()[0])
    print('drugScore: ', df_pockets_ds_05_wpr_cluster['drugScore'].tolist()[0])