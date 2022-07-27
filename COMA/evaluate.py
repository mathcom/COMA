import tqdm
import numpy as np
import pandas as pd
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from COMA.properties import FastTanimotoOneToBulk


def evaluate_metric(df_generated, smiles_train_high, num_decode=20, threshold_pro=0.0, threshold_improve=0.0, list_threshold_sim=[0.4]):
    ## Init
    df_metrics = pd.DataFrame(0., columns=["VALID_RATIO", "PROPERTY",  "IMPROVEMENT",  "SIMILARITY", "NOVELTY", "DIVERSITY"], index=[0])
    df_sr = pd.DataFrame(0., columns=["SR_PROP", "SR_IMPR", "SR_PROP_WO_NOVELTY", "SR_IMPR_WO_NOVELTY"], index=list_threshold_sim)
    df_sr.index.name = "THRESHOLD_SIMILARITY"
    
    ## Assert
    num_molecules = len(df_generated) // num_decode
    assert len(df_generated) % num_decode == 0
    
    ## Do
    for i in range(0, len(df_generated), num_decode):
        sources = set([x for x in df_generated.iloc[i:i+num_decode, 0]])
        assert len(sources) == 1
        
        ###################################
        ## Metric 1) Validity
        ###################################
        targets_valid = [(tar, sim, prop_tar, prop_src) for src, tar, sim, prop_tar, prop_src in df_generated.iloc[i:i+num_decode,:].values if 1 > sim > 0 and prop_tar > 0]
        if len(targets_valid) > 0:
            df_metrics.loc[0, "VALID_RATIO"] += 1
            
        ###################################
        ## Metric 2) Property
        ###################################
        targets_valid_prop = [prop_tar for _, _, prop_tar, _ in targets_valid]
        if len(targets_valid_prop) > 0:
            df_metrics.loc[0, "PROPERTY"] += np.mean(targets_valid_prop)
    
        ###################################
        ## Metric 2) Improvement
        ###################################
        targets_valid_impr = [prop_tar - prop_src for _, _, prop_tar, prop_src in targets_valid]
        if len(targets_valid_impr) > 0:
            df_metrics.loc[0, "IMPROVEMENT"] += np.mean(targets_valid_impr)
    
        ###################################
        ## Metric 3) Similarity
        ###################################
        targets_valid_sim = [sim for _, sim, _, _ in targets_valid]
        if len(targets_valid_sim) > 0:
            df_metrics.loc[0, "SIMILARITY"] += np.mean(targets_valid_sim)
    
        ###################################
        ## Metric 4) Novelty
        ###################################
        targets_novel = [(tar, sim, prop_tar, prop_src) for tar, sim, prop_tar, prop_src in targets_valid if tar not in smiles_train_high]
        if len(targets_novel) > 0:
            df_metrics.loc[0, "NOVELTY"] += 1
            
        ###################################
        ## Metric 5) Diversity
        ###################################
        if len(targets_valid) > 1:
            calc_bulk_sim = FastTanimotoOneToBulk([x[0] for x in targets_valid])
            similarity_between_targets = []            
            for j in range(len(targets_valid)):
                div = calc_bulk_sim(targets_valid[j][0])
                similarity_between_targets += div[:j-1].tolist() + div[j+1:].tolist()
            df_metrics.loc[0, "DIVERSITY"] += 1. - np.mean(similarity_between_targets)
            
        ###################################
        ## Metric 6) Success Rates
        ###################################
        for threshold_sim in list_threshold_sim:
            ## Property-based success rate
            targets_success = [(tar, sim, prop_tar, prop_src) for tar, sim, prop_tar, prop_src in targets_novel if sim >= threshold_sim and prop_tar >= threshold_pro]
            if len(targets_success) > 0:
                df_sr.loc[threshold_sim, 'SR_PROP'] += 1
                
            ## Improvement-based success rate
            targets_success = [(tar, sim, prop_tar, prop_src) for tar, sim, prop_tar, prop_src in targets_novel if sim >= threshold_sim and prop_tar - prop_src > threshold_improve]
            if len(targets_success) > 0:
                df_sr.loc[threshold_sim, 'SR_IMPR'] += 1
                
            ## Property-based success rate without novelty constraint
            targets_success = [(tar, sim, prop_tar, prop_src) for tar, sim, prop_tar, prop_src in targets_valid if sim >= threshold_sim and prop_tar >= threshold_pro]
            if len(targets_success) > 0:
                df_sr.loc[threshold_sim, 'SR_PROP_WO_NOVELTY'] += 1
                
            ## Improvement-based success rate without novelty constraint
            targets_success = [(tar, sim, prop_tar, prop_src) for tar, sim, prop_tar, prop_src in targets_valid if sim >= threshold_sim and prop_tar - prop_src > threshold_improve]
            if len(targets_success) > 0:
                df_sr.loc[threshold_sim, 'SR_IMPR_WO_NOVELTY'] += 1
            
    
    ###################################
    ## Final average
    ###################################
    df_metrics.iloc[0,:] = df_metrics.iloc[0,:] / num_molecules
    df_sr.iloc[:,:] = df_sr.iloc[:,:] / num_molecules

    return {'metrics':df_metrics, 'success_rate':df_sr.reset_index()}


def evaluate_metric_validation(df_generated, num_decode=20):
    metrics = {"VALID_RATIO":0.,
               "AVERAGE_PROPERTY":0.,
               "AVERAGE_SIMILARITY":0.}               
    
    num_molecules = len(df_generated) // num_decode
    assert len(df_generated) % num_decode == 0
    
    for i in range(0, len(df_generated), num_decode):
        sources = set([x for x in df_generated.iloc[i:i+num_decode, 0]])
        assert len(sources) == 1
        
        ###################################
        ## Metric 1) Validity
        ###################################
        targets_valid = [(tar,sim,prop) for _,tar,sim,prop in df_generated.iloc[i:i+num_decode,:].values if 1 > sim > 0 and prop > 0]
        if len(targets_valid) > 0:
            metrics["VALID_RATIO"] += 1
            
        ###################################
        ## Metric 2) Property
        ###################################
        targets_valid_prop = [prop for _, _, prop in targets_valid]
        if len(targets_valid_prop) > 0:
            metrics["AVERAGE_PROPERTY"] += np.mean(targets_valid_prop)
    
        ###################################
        ## Metric 3) Similarity
        ###################################
        targets_valid_sim = [sim for _, sim, _ in targets_valid]
        if len(targets_valid_sim) > 0:
            metrics["AVERAGE_SIMILARITY"] += np.mean(targets_valid_sim)
            
    ###################################
    ## Final average
    ###################################
    metrics["VALID_RATIO"]        /= num_molecules
    metrics["AVERAGE_PROPERTY"]   /= num_molecules
    metrics["AVERAGE_SIMILARITY"] /= num_molecules
  
    df_metrics = pd.Series(metrics).to_frame()
    return df_metrics
