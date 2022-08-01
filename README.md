# COMA: efficient structure-constrained molecular generation using COnstractive and MArgin losses

<img src="figs/overview_of_COMA.png" alt="thumbnail" width="600px" />

We propose COMA, a structure-constrained molecular generation model with metric learning and reinforcement learning, to achieve property improvement and high structural similarity performance at once.

COMA translates a source molecule into a novel molecule having desirable chemical properties but not large changes of molecular structures.

COMA requires simplified molecular-input line-entry system (SMILES) strings as inputs.

For more detail, please refer to Choi, et. al. "Efficient Structure-constrained Molecular Generation using Constarctive and Margin losses" (under review)


* Latest update: 28 July 2022

--------------------------------------------------------------------------------------------
## SYSTEM REQUIERMENTS: 

- MTMR requires system memory larger than 8GB.

- (if GPU is available) MTMR requires GPU memory larger than 8GB.


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

- Before running tutorials, an user should decompress the compressed files: DATA/{name}.tar.gz

- The following commands are for decompression:

```bash
cd DATA
tar -xzvf drd2.tar.gz
tar -xzvf qed.tar.gz
tar -xzvf logp04.tar.gz
tar -xzvf logp06.tar.gz
cd ..
```

- Due to the large size of the sorafenib dataset, please contact me if you need the dataset.
  

--------------------------------------------------------------------------------------------
## Tutorials:

- We provide several jupyter-notebooks for tutorials.
  - 1_pretraining.ipynb
  - 2_finetuning.ipynb
  - 3_latent_space_analysis.ipynb
  - 4_translation.ipynb
  - 5_evaluation.ipynb

- These tutorial files are available for reproducibility purposes.

- An user can open them using the following commands:

```bash
conda activate coma
jupyter notebook

~ run tutorial ~

conda deactivate
```


--------------------------------------------------------------------------------------------
## Contact:

- Email: mathcombio@yonsei.ac.kr


--------------------------------------------------------------------------------------------
