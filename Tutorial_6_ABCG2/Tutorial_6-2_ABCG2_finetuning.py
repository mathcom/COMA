import os
import sys
import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if HOME_PATH is not in sys.path:
    sys.path = [HOME_PATH] + sys.path
    
from COMA.dataset import ValidationSmilesDataset
from COMA.vae import SmilesAutoencoder, ReplayBufferDataset
from COMA.properties import similarity, affinity


class RewardFunction(object):
    def __init__(self, similarity_ft, scoring_ft, scoring_ft_2):
        super(RewardFunction, self).__init__()
        self.similarity_ft = similarity_ft
        self.scoring_ft = scoring_ft
        self.scoring_ft_2 = scoring_ft_2
        self.threshold_similarity = 0.4
        self.threshold_property = 4.989
        self.threshold_property_2 = 6.235
        
    def __call__(self, smi_src, smi_tar):
        score_pro = self.scoring_ft(smi_tar)
        score_pro_2 = self.scoring_ft_2(smi_tar)
        score_sim = self.similarity_ft(smi_src, smi_tar)
        
        if score_sim > self.threshold_similarity and score_pro_2 > self.threshold_property_2:
            reward = max(1 - (score_pro / self.threshold_property), 0.)
            return (reward, score_sim, score_pro)
        else:
            return (0., score_sim, score_pro)


if __name__=="__main__":
    ###################################
    ## 0. Environment Setting
    ###################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    PROPERTY_NAME = "ABCG2"
    SCORING_FT = affinity("MSSSNVEVFIPVSQGNTNGFPATASNDLKAFTEGAVLSFHNICYRVKLKSGFLPCRKPVEKEILSNINGIMKPGLNAILGPTGGGKSSLLDVLAARKDPSGLSGDVLINGAPRPANFKCNSGYVVQDDVVMGTLTVRENLQFSAALRLATTMTNHEKNERINRVIQELGLDKVADSKVGTQFIRGVSGGERKRTSIGMELITDPSILFLDEPTTGLDSSTANAVLLLLKRMSKQGRTIIFSIHQPRYSIFKLFDSLTLLASGRLMFHGPAQEALGYFESAGYHCEAYNNPADFFLDIINGDSTAVALNREEDFKATEIIEPSKQDKPLIEKLAEIYVNSSFYKETKAELHQLSGGEKKKKITVFKEISYTTSFCHQLRWVSKRSFKNLLGNPQASIAQIIVTVVLGLVIGAIYFGLKNDSTGIQNRAGVLFFLTTNQCFSSVSAVELFVVEKKLFIHEYISGYYRVSSYFLGKLLSDLLPMRMLPSIIFTCIVYFMLGLKPKADAFFVMMFTLMMVAYSASSMALAIAAGQSVVSVATLLMTICFVFMMIFSGLLVNLTTIASWLSWLQYFSIPRYGFTALQHNEFLGQNFCPGLNATGNNPCNYATCTGEEYLVKQGIDLSPWGLWKNHVALACMIVIFLTIAYLKLLFLKKYS", device)
    SCORING_FT_2 = affinity("MAALSGGGGGGAEPGQALFNGDMEPEAGAGAGAAASSAADPAIPEEVWNIKQMIKLTQEHIEALLDKFGGEHNPPSIYLEAYEEYTSKLDALQQREQQLLESLGNGTDFSVSSSASMDTVTSSSSSSLSVLPSSLSVFQNPTDVARSNPKSPQKPIVRVFLPNKQRTVVPARCGVTVRDSLKKALMMRGLIPECCAVYRIQDGEKKPIGWDTDISWLTGEELHVEVLENVPLTTHNFVRKTFFTLAFCDFCRKLLFQGFRCQTCGYKFHQRCSTEVPLMCVNYDQLDLLFVSKFFEHHPIPQEEASLAETALTSGSSPSAPASDSIGPQILTSPSPSKSIPIPQPFRPADEDHRNQFGQRDRSSSAPNVHINTIEPVNIDDLIRDQGFRGDGGSTTGLSATPPASLPGSLTNVKALQKSPGPQRERKSSSSSEDRNRMKTLGRRDSSDDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDVAVKMLNVTAPTPQQLQAFKNEVGVLRKTRHVNILLFMGYSTKPQLAIVTQWCEGSSLYHHLHIIETKFEMIKLIDIARQTAQGMDYLHAKSIIHRDLKSNNIFLHEDLTVKIGDFGLATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQILASIELLARSLPKIHRSASEPSLNRAGFQTEDFSLYACASPKTPIQAGGYGAFPVH", device)
    
    ## Directory
    input_dir = f"outputs_6-1_{PROPERTY_NAME}_pretraining"
    output_dir = f"outputs_6-2_{PROPERTY_NAME}_finetuning"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    ###################################
    ## 1. Dataset
    ###################################
    print("[INFO] 1. Dataset")   
    filepath_char2idx = os.path.join(input_dir, "char2idx.csv")
    filepath_inputs_train_src = os.path.join("data", PROPERTY_NAME, "rdkit_train_src.csv")
    dataset_train_src = ValidationSmilesDataset(filepath_inputs_train_src, filepath_char2idx, device=device)
    dataset_train_src.save_char2idx(os.path.join(output_dir, "char2idx.csv"))

    ###################################
    ## 2. Model
    ###################################
    print("[INFO] 2. Model")
    filepath_pretrain_configs = os.path.join(input_dir, "configs.csv")
    filepath_pretrain_ckpt = os.path.join(input_dir, "SmilesAutoencoder.pt")
    
    ## Model configuration
    model_configs = {"hidden_size"    :None,
                     "latent_size"    :None,
                     "num_layers"     :None,
                     "vocab_size"     :None,
                     "sos_idx"        :None,
                     "eos_idx"        :None,
                     "pad_idx"        :None,
                     "device"         :device,
                     "filepath_config":filepath_pretrain_configs}
    ## Model initialization
    generator = SmilesAutoencoder(**model_configs)
    ## Load pretrained model
    generator.load_model(filepath_pretrain_ckpt)
    ## save configuration
    generator.save_config(os.path.join(output_dir, "configs.csv"))
    
    ###################################
    ## 3. Reinforcement Learning
    ###################################
    reward_ft = RewardFunction(similarity, SCORING_FT, SCORING_FT_2)
    
    ###################################
    ## 4. Train
    ###################################
    print("[INFO] 3. Train")
    batch_size = 1000
    buffer_size = 2000
    sample_size = 50
    total_steps = 4000
    
    step = 0
    history = []
    replay_buffer = ReplayBufferDataset()
    start_time = time.time()
    while step < total_steps:
        ## Fill the buffer
        for batch in DataLoader(dataset_train_src, batch_size=1, shuffle=True, drop_last=False, pin_memory=False):
            ## input data preprocess
            single_encode = dataset_train_src.encode(batch["smiles_s"], batch["length_s"].max()) # single_encode.shape = (1, seq)
            single_length = batch["length_s"] # single_length.shape = (1, )
            
            ## sample size
            batch_inp_encode = single_encode.repeat(sample_size, 1) # batch_inp_encode.shape = (sample, seq)
            batch_inp_length = single_length.repeat(sample_size) # batch_inp_length.shape = (sample, )
            
            ## generate target data
            batch_out_encode = generator.predict(batch_inp_encode, batch_inp_length) # batch_out_encode.shape = (sample, max_seqlen)
            batch_tar_smiles = [dataset_train_src.sos_char + smi + dataset_train_src.eos_char for smi in dataset_train_src.decode(batch_out_encode)] # len(batch_tar_smiles) = batch
            
            ## reward
            smi_src = batch["smiles_s"][0]
            batch_properties = np.array([reward_ft(smi_src[1:-1], smi_tar[1:-1]) for smi_tar in batch_tar_smiles])
            
            ## reward normalization
            batch_reward = batch_properties[:,0]
            batch_properties[:,0] = batch_reward / (np.max(batch_reward) + 1e-8)
            
            ## append
            sample_buffer = []
            for smi_tar, (rew_tar, sim_tar, pro_tar) in zip(batch_tar_smiles, batch_properties):
                if rew_tar > 0:
                    replay_buffer.push(smi_src, smi_tar, rew_tar, sim_tar, pro_tar)

            ## is buffer full?
            if len(replay_buffer) >= buffer_size:
                break
            
        ## buffer statistics
        avg_reward, avg_similarity, avg_properties = replay_buffer.stats()
            
        ## Replay episodes
        for batch in DataLoader(replay_buffer, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False):
            batch_encode_src = dataset_train_src.encode(batch["smiles_src"], batch["length_src"].max())
            batch_length_src = batch["length_src"]
            batch_encode_tar = dataset_train_src.encode(batch["smiles_tar"], batch["length_tar"].max())
            batch_length_tar = batch["length_tar"]
            batch_reward = batch["reward"].to(device)
            ## policy gradient
            rl_loss = generator.partial_policy_gradient(batch_encode_src, batch_length_src, batch_encode_tar, batch_length_tar, batch_reward)
            ## buffer update
            replay_buffer.pop()
            break

        ## log
        execution_sec = time.time() - start_time
        execution_min = execution_sec // 60
        if step % 10 == 0:
            log = f"[{step:06d}/{total_steps:06d}]"
            log += f"  loss: {rl_loss:.3f}"
            log += f"  reward: {avg_reward:.3f}"
            log += f"  score(similarity): {avg_similarity:.3f}"
            log += f"  score({PROPERTY_NAME}): {avg_properties:.3f}"
            log += f"  ({execution_min} min)"
            print(log)
            
            ## model save
            filepath_model = os.path.join(output_dir, f"SmilesAutoencoder_{step:06d}.pt")
            generator.save_model(filepath_model)
            
            ## history
            filepath_history = os.path.join(output_dir, "history.csv")
            df_history = pd.DataFrame(history, columns=["LOSS",
                                                        "REWARD",
                                                        "SIMILARITY",
                                                        "PROPERTY",
                                                        "EXEC_TIME_SEC"])
            df_history.to_csv(filepath_history, index=False)
            
        ## history
        history.append((rl_loss,
                        avg_reward,
                        avg_similarity,
                        avg_properties,
                        execution_sec))
            
        ## termination for training
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
    df_history = pd.DataFrame(history, columns=["LOSS",
                                                "REWARD",
                                                "SIMILARITY",
                                                "PROPERTY",
                                                "EXEC_TIME_SEC"])
    df_history.to_csv(filepath_history, index=False)
    