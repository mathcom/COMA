import os
import sys
import pandas as pd
import time
import torch
import tqdm
from torch.utils.data import DataLoader
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if HOME_PATH is not in sys.path:
    sys.path = [HOME_PATH] + sys.path
    
from COMA.dataset import ValidationSmilesDataset
from COMA.vae import SmilesAutoencoder
from COMA.properties import affinity, similarity


def run_on_dataset(generator, dataset_test, use_cuda):
    K = 20 # repetition count of translation
    generated = [] # initialize a list of outputs

    for batch in tqdm.tqdm(DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False, pin_memory=use_cuda)):
        batch_smiles = dataset_test.encode(batch["smiles_s"], batch["length_s"].max())
        batch_length = batch["length_s"]
        ## translation
        for _ in range(K):
            seq = generator.predict(batch_smiles, batch_length)
            smi = dataset_test.decode(seq)[0] # assumption: batch_size=1
            if MolFromSmiles(smi) is not None:
                generated.append((batch["smiles_s"][0][1:-1], smi))
            else:
                generated.append((batch["smiles_s"][0][1:-1], "None"))
            
    df_generated = pd.DataFrame.from_records(generated)
    return df_generated

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
    input_dir = f"outputs_6-2_{PROPERTY_NAME}_finetuning"
    output_dir = f"outputs_6-3_{PROPERTY_NAME}_translation"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    ###################################
    ## 1. Dataset
    ###################################
    print("[INFO] 1. Dataset")   
    filepath_char2idx = os.path.join(input_dir, "char2idx.csv")
    filepath_inputs_test = os.path.join("data", PROPERTY_NAME, "rdkit_test.txt")
    dataset_test = SmilesValidationDataset(filepath_inputs_test, filepath_char2idx, device=device)
    
    filepath_inputs_sorafenib = os.path.join("data", PROPERTY_NAME, "rdkit_sorafenib.txt")
    dataset_sorafenib = SmilesValidationDataset(filepath_inputs_sorafenib, filepath_char2idx, device=device)

    ###################################
    ## 2. Model
    ###################################
    print("[INFO] 2. Model")
    filepath_pretrain_configs = os.path.join(input_dir, "configs.csv")
    filepath_pretrain_ckpt = os.path.join(input_dir, "SmilesAutoencoder_002000.pt")
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
    
    ###################################
    ## 3. Run on Test
    ###################################
    df_generated_test = run_on_dataset(generator, dataset_test, use_cuda)
    filepath_generated_test = os.path.join(output_dir, "results_test_whole.csv")
    df_generated_test.to_csv(filepath_generated_test, index=False, header=None)

    ###################################
    ## 4. Run on Sorafenib
    ###################################
    df_generated_sorafenib = run_on_dataset(generator, dataset_sorafenib, use_cuda)
    filepath_generated_sorafenib = os.path.join(output_dir, "results_sorafenib.csv")
    df_generated_sorafenib.to_csv(filepath_generated_sorafenib, index=False, header=None)
