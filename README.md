# structbio_workflows

This is a repository of structural bioinformatics workflows, designed for tasks related to topics such as structure-based drug design, molecular dynamics simulation setup and analysis.

The workflows include Python scripts which can be run from the Linux terminal. 

Please note that although the workflows are tested prior to their deposition here, they have not been designed with end users in mind. We only provide these for convenience to those who intend to perform similar tasks for their own projects. 

## List of workflows:

### dss.py

### dss_analysis.py
A workflow that parses binding pockets predicted by DoGSiteScorer 2 from a biomolecular simulation trajectory/structural ensemble, and generates a clustermap showing residues forming different pockets on protein surface. 

This clustermap provides a visualization of potential druggable binding pockets and their persistence within the content of native state dynamics of the target protein. This may be useful for selection of potentially allosteric or cryptic binding pockets on protein structures.

### md_traj_analysis.py

### pandora.py

### frustratometer.py

### smina_screen.py

### retrieve_struct_batch.py

### pm_screen.py

## How to use the workflows?

Clone the repository, open the file using Jupyter Notebook/JupyterLab or your favorite IDE (e.g. VSCode), install any dependencies required (using either pip or conda), modify the code as needed.

## Questions and comments

Please reach out to me (Onur Serçinoğlu) in case you have any questions or comments regarding these workflows. 
