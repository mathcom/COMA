import os
import sys
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if HOME_PATH is not in sys.path:
    sys.path = [HOME_PATH] + sys.path
    
from COMA.dataset import TrainingSmilesDataset, ValidationSmilesDataset
from COMA.vae import SmilesAutoencoder, AnnealingScheduler


if __name__=="__main__":
    ###################################
    ## 0. Environment Setting
    ###################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    PROPERTY_NAME = "ABCG2"

    output_dir = f"outputs_6-1_{PROPERTY_NAME}_pretraining"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    ###################################
    ## 1. Dataset
    ###################################
    print("[INFO] 1. Dataset")
    input_dir = os.path.join("data", PROPERTY_NAME)
    filepath_train = os.path.join(input_dir, "rdkit_train_triplet.txt")
    filepath_char2idx = os.path.join(output_dir, "char2idx.csv")
    
    dataset = TrainingSmilesDataset(filepath_train, device=device)
    dataset.save_char2idx(filepath_char2idx)
    
    ###################################
    ## 2. Model
    ###################################
    print("[INFO] 2. Model")
    ## model configuration
    model_configs = {"hidden_size":128,
                     "latent_size":128,
                     "num_layers" :2,
                     "vocab_size" :dataset.vocab_size,
                     "sos_idx"    :dataset.sos_idx,
                     "eos_idx"    :dataset.eos_idx,
                     "pad_idx"    :dataset.pad_idx,
                     "device"     :device
                    }
                    
    ## model initialization
    generator = SmilesAutoencoder(**model_configs)
    
    ## save configuration
    filepath_configs = os.path.join(output_dir, "configs.csv")
    generator.save_config(filepath_configs)
    
    ###################################
    ## 3. Train
    ###################################
    print("[INFO] 3. Train")
    batch_size = 100
    total_steps = 100000
    
    ## Scheduler
    calc_beta = AnnealingScheduler(total_steps)
    calc_gamma = AnnealingScheduler(total_steps)
    
    ## Initialization for training
    step = 0
    history = []
    start_time = time.time()
    
    ## Train
    while step < total_steps:
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=use_cuda):
            ## data preprocess
            batch_smiles_A = dataset.encode(batch["smiles_s"], batch["length_s"].max())
            batch_length_A = batch["length_s"]
            batch_smiles_B = dataset.encode(batch["smiles_t"], batch["length_t"].max())
            batch_length_B = batch["length_t"]
            batch_smiles_C = dataset.encode(batch["smiles_n"], batch["length_n"].max())
            batch_length_C = batch["length_n"]
            batch_inputs = (batch_smiles_A, batch_length_A, batch_smiles_B, batch_length_B, batch_smiles_C, batch_length_C)
            
            ## fit
            beta = calc_beta(step)
            gamma = calc_gamma(step)
            batch_loss = generator.partial_fit(*batch_inputs, beta=beta, gamma=gamma)
            
            ## loss for training data
            loss_train             = batch_loss[0]
            loss_recon_A_train     = batch_loss[1] # L(src)
            loss_recon_B_train     = batch_loss[2] # L(tar)
            loss_recon_C_train     = batch_loss[3] # L(neg)
            loss_contractive_train = batch_loss[4] # L(src,tar)
            loss_margin_train      = batch_loss[5] # L(src,neg) + L(tar,neg)
            
            ## log
            execution_sec = time.time() - start_time
            execution_min = execution_sec // 60
            if step % 1000 == 0:
                log = f"[{step:06d}/{total_steps:06d}]"
                log += f"  loss(tr): {loss_train:.3f}"
                log += f"  loss_recon_src(tr): {loss_recon_A_train:.3f}"
                log += f"  loss_recon_tar(tr): {loss_recon_B_train:.3f}"
                log += f"  loss_recon_neg(tr): {loss_recon_C_train:.3f}"
                log += f"  loss_contractive(tr): {loss_contractive_train:.3f}"
                log += f"  loss_margin(tr): {loss_margin_train:.3f}"
                log += f"  beta: {beta:.3f}"
                log += f"  gamma: {gamma:.3f}"
                log += f"  ({execution_min} min)"
                print(log)
                
                ## model save
                filepath_model = os.path.join(output_dir, f"SmilesAutoencoder_{step:06d}.pt")
                generator.save_model(filepath_model)
                
            ## history
            history.append((loss_train,
                            loss_recon_A_train,
                            loss_recon_B_train,
                            loss_recon_C_train,
                            loss_contractive_train,
                            loss_margin_train,
                            beta,
                            gamma,
                            execution_sec))
            
            ## termination
            if step >= total_steps:
                break
                
            step += 1
            
    ###################################
    ## 4. Save
    ###################################
    print("[INFO] 4. Save")
    filepath_model = os.path.join(output_dir, "SmilesAutoencoder.pt")
    generator.save_model(filepath_model)
    
    ## history
    filepath_history = os.path.join(output_dir, "history.csv")
    df_history = pd.DataFrame(history, columns=["LOSS_TOTAL",
                                                "LOSS_RECONSTRUCTION_SOURCE",
                                                "LOSS_RECONSTRUCTION_TARGET",
                                                "LOSS_RECONSTRUCTION_NEGATIVE",
                                                "LOSS_CONTRACTIVE",
                                                "LOSS_MARGIN",
                                                "BETA",
                                                "GAMMA",
                                                "EXEC_TIME_SEC"])
    df_history.to_csv(filepath_history, index=False)
    
    
