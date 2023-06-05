import pandas
import re
from openbabel import pybel

from opencadd.structure.core import Structure
from opencadd.io.dataframe import DataFrame

# filter warnings
warnings.filterwarnings("ignore")
ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.SetOutputLevel(0)

def pdb_to_pdbqt(pdb_path, pdbqt_path, pH=7.4):
    """
    Convert a PDB file to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    pdb_path: str or pathlib.Path
        Path to input PDB file.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    # add hydrogens at given pH
    #molecule.OBMol.CorrectForPH(pH)
    #molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return

#%%
# convert protein to PDBQT format
receptor_fn = "frame_2152"
pdb_to_pdbqt(receptor_fn+".pdb", receptor_fn+".pdbqt")

#%%
structure_df = DataFrame.from_file("frame_2152_res_P_2_1.pdb")
positions = np.array([structure_df["atom.x"].values,structure_df["atom.y"].values,structure_df["atom.z"].values])
pocket_center = list(map(str,((np.max(positions,axis=1) + np.min(positions,axis=1)) / 2)))
pocket_size = list(map(str,((np.max(positions,axis=1) - np.min(positions,axis=1)) + 5)))

subprocess.call(f"smina -r {receptor_fn}.pdbqt -l e-Drug3D_2056.sdf \
    --center_x {pocket_center[0]} --center_y {pocket_center[1]} --center_z {pocket_center[2]} \
    --size_x {pocket_size[0]} --size_y {pocket_size[1]} --size_z {pocket_size[2]} --out screen_P_2_1_2023_05_19.sdf \
    --num_modes 20 --exhaustiveness 8 --cpu 12 --log screen_P_2_1_2023_05_19.log",shell=True)

smina_output_f = "screen_2023_05_19.sdf"
 
ligands = []
with open(smina_output_f,"r") as smina_output:
	lines = smina_output.readlines()
	lines = [line.rstrip('\n') for line in lines]
	ligands.append(lines[0])
	read_flag = 0
	for line in lines:
 
		if read_flag:
			ligands.append(line)
 
		if not line.startswith("$$$$"):
			read_flag = 0
			continue
		else:
			read_flag = 1
			continue

smina_log_f = "screen_2023_05_19.log"
 
df_results = pandas.DataFrame(columns=["Pose","Score","RMSD_lb","RMSD_ub"])
with open(smina_log_f,"r") as smina_output:
	lines = smina_output.readlines()
	lines = [line.rstrip('\n') for line in lines]
 
	read_flag = 0
	for line in lines:
 
		if line.startswith("Using random seed"):
			read_flag = 0
			continue
 
		if read_flag:
			line2 = line.split(' ')
			line2 = [el for el in line2 if el]
			line2 = list(map(float,line2))
			df_results.loc[len(df_results),:] = line2
 
		if line.startswith("-----"):
			read_flag = 1
			continue
 
df_results['Ligand'] = ligands
df_results.to_csv('df_results_2023_05_19.csv')
print(df_results)