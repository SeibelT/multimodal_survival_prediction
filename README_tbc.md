# Survival Prediction from Genomic and Histological Data 
This Repository contains the code which was used for my Master thesis....
## Table of Contents
1. [Background](#background)
2. [Late Fusion](#late-fusion)
3. [Middle Fusion](#middle-fusion)
4. [Early Fusion](#early-fusion)
5. [How to Use](#how-to-use)
    - 5.1 [Prepare CSV](#sprep_csv)
    - 5.2 [sweep](#sweep)
    - 5.3 [Feature Encoder Finetuning](#featureenc)

## Background
This Project uses Porpoise *###insert link###* 
as a baseline and therefor contains the juupyter notebook which disects the concept into its separate parts.  

## Late Fusion
Based on the paper and the given repository of porpoise, 
their model is recreated. Multiple  parameter sweep with k-fold crossvalidation were applied to analyse the training behaviour of the unimodal and multimodal models.  

## Middle Fusion
To shift the fusion to an earlier layer of the model, a further k-fold crossvalidation hyperparameter sweep was done called PrePorpoise. Comparing it to the Porpoise model, it seems like the results benefit from an earlier fusion.

## Early Fusion
For an even earlier fusion, multiple models were trained to apply multimodality and task transfer fine tuning on the feature encoder level for the tiles of the WSI.  #TODO 

## How to Use
This section provides a guide on how to use the main functions implemented in this repository for recreation.

### ***prep_csv***
**Description:** use the function within utils to  create the csv file that contains genomic data, filepath, and  further patient data for training.  #TODO

**Usage:**
1. traintest:
    * ``python create_ds.py --split trainitest --frac 0.8 --link1 <> --link2 <> --bins 4``
2. k fold:
    * ``python create_ds.py --split kfold --kfolds 5 --link1 <> --link2 <> --bins 4``

### ***sweep***
**Description:** to run a wandb sweep with a  k-fold crossvalidation the following steps have to be applied

**Usage:**
0. adapt folder paths within code # TODO add fix paths to yaml files
1. create sweep file and add to sweeps folder
2. wandb sweep sweeps/<sweepfilename.yaml>
    * this returns an agent name such as: 
    * `wandb agent wandbname/projectname/ab123cdef4`

3. run agent within sbatch file:
    * `sbatch sweeps.sh wandb agent wandbname/projectname/ab123cdef4`
    * the commandline can be executed multiple times for multiple agents, each doing one sweep setting
    
### ***featureenc***
**Description:** #TODO
**Usage:** #TODO