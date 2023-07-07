# COMA: efficient structure-constrained molecular generation using COnstractive and MArgin losses

- Latest update: 07 July 2023

<img src="figs/overview_of_COMA.png" alt="thumbnail" width="600px" />

This repository is for COMA, a structure-constrained molecular generative model.

For a given source molecule, COMA generates a novel molecule with more improved chemical properties by making a small modification on the source structure.

To achieve property improvement and high structural similarity simultaneously, COMA exploits reinforcement learning and metric learning.

For more detail, please refer to J. Choi, S. Seo, and S. Park. **COMA: efficient structure-constrained molecular generation using contractive and margin losses**. *J Cheminform* 15, 8 (2023). https://doi.org/10.1186/s13321-023-00679-y


--------------------------------------------------------------------------------------------
## SYSTEM REQUIERMENTS: 

- (If GPU is available) COMA may require GPU memory larger than 6GB.
  - Available cudatoolkit versions: 10.2, 11.1, and 11.3

- **COMA is only for Python 3.7**

--------------------------------------------------------------------------------------------
## Installation:

- We recommend to install via Anaconda (https://www.anaconda.com/)

- After installing Anaconda, please create a conda environment with the following commands:

```bash
git clone https://github.com/mathcom/COMA.git
cd COMA
conda env create -f environment.yml
```

--------------------------------------------------------------------------------------------
## Data:

- Before running tutorials, an user should decompress the compressed files: data/{name}.tar.gz

- The following commands are for decompression:

```bash
cd data
tar -xzvf drd2.tar.gz
tar -xzvf qed.tar.gz
tar -xzvf logp04.tar.gz
tar -xzvf logp06.tar.gz
cd ..
```

- After decompressing, an user can find the following files and is ready to run the provided scripts.
  - rdkit_test.txt
  - rdkit_train_pairs.txt
  - rdkit_train_src.txt
  - rdkit_train_tar.txt
  - rdkit_train_triplet.txt
  - rdkit_valid.txt
  
- The details of how to create a triplet dataset are described in the Algorithm S3 of our paper.
  

--------------------------------------------------------------------------------------------
## Scripts:

- We provide several jupyter-notebooks, which are available for reproducibility.
  - 1_pretraining.ipynb
  - 2_finetuning.ipynb
  - 3_latent_space_analysis.ipynb
  - 4_generation.ipynb
  - 5_evaluation.ipynb
  - 6_drawing_molecules.ipynb

- An user can open them using the following commands:

```bash
conda activate coma
jupyter notebook

~ run tutorial ~

conda deactivate
```


--------------------------------------------------------------------------------------------
## Source codes:

- The source codes of COMA are available at https://github.com/mathcom/mol-coma


--------------------------------------------------------------------------------------------
## Contact:

- Email: mathcombio@yonsei.ac.kr


--------------------------------------------------------------------------------------------
