"""
Written by Jonghwan Choi at 12 July 2022
https://github.com/mathcom/MTMR

The triplet loss was defined in:
Ishfaq, Haque, A. Hoogi, and D. Rubin. "TVAE: Deep metric learning approach for variational autoencoder."
Workshop at International Conference on Learning Representations (ICLRW). Vol. 32. 2018.
"""
import math
import numpy as np
import pandas as pd
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from COMA.evaluate import evaluate_metric_validation
from COMA.properties import similarity


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, pad_idx, num_layers, dropout, device=None):
        super(SmilesEncoder, self).__init__()
        ## params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cpu') if device is None else device
        
        ## special tokens
        self.pad_idx = pad_idx
        
        ## Neural Network
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size, padding_idx=self.pad_idx)
        self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2mean = nn.Linear(self.num_layers * 2 * self.hidden_size, self.latent_size)
        self.hidden2logvar = nn.Linear(self.num_layers * 2 * self.hidden_size, self.latent_size)


    def forward(self, inps, lens):
        '''
        Params
        ------
        inps.shape = (batch, maxseqlen)
        lens.shape = (batch,)
        '''
        batch_size = inps.size(0)
        
        ## Sorting by seqlen
        sorted_seqlen, sorted_idx = torch.sort(lens, descending=True) # sorted_seqlen.shape = (batch,), sorted_idx.shape = (batch,)
        sorted_inps = inps[sorted_idx] # sorted_inps.shape = (batch, maxseqlen)
        
        ## Packing for encoder
        inps_emb = self.embedding(sorted_inps) # inps_emb.shape = (batch, maxseqlen, hidden)
        packed_inps = rnn_utils.pack_padded_sequence(inps_emb, sorted_seqlen.data.tolist(), batch_first=True)
        
        ## RNN
        _, sorted_hiddens = self.rnn(packed_inps) # sorted_hiddens.shape = (numlayer * 2, batch, hidden)
        sorted_hiddens = sorted_hiddens.transpose(0,1).contiguous() # sorted_hiddens.shape = (batch, numlayer * 2, hidden)
        sorted_hiddens = sorted_hiddens.view(batch_size, -1) # sorted_hiddens.shape = (batch, numlayer * 2 * hidden)
        
        ## Latent vector
        sorted_mean = self.hidden2mean(sorted_hiddens) # sorted_mean.shape = (batch, latent)
        sorted_logvar = self.hidden2logvar(sorted_hiddens) # sorted_logvar.shape = (batch, latent)
        
        ## Reordering
        mean, logvar = self.reordering(sorted_mean, sorted_logvar, sorted_idx)
        return mean, logvar
        
        
    def sampling(self, mean, logvar):
        '''
        Params
        ------
        mean.shape = (batch, latent)
        logvar.shape = (batch, latent)
        '''
        batch_size = mean.size(0)
        std = torch.exp(0.5 * logvar) # std.shape = (batch, latent)
        epsilon = torch.randn([batch_size, self.latent_size], device=self.device) # epsilon.shape = (batch, latent)
        z = epsilon * std + mean # z.shape = (batch, latent)
        return z
        
        
    def reordering(self, sorted_mean, sorted_logvar, sorted_idx):
        '''
        Params
        ------
        sorted_mean.shape = (batch, latent)
        sorted_logvar.shape = (batch, latent)
        sorted_idx = (batch, )
        '''
        _, original_idx = torch.sort(sorted_idx, descending=False) # original_idx.shape = (batch, )
        mean = sorted_mean[original_idx] # mean.shape = (batch, latent)
        logvar = sorted_logvar[original_idx] # logvar.shape = (batch, latent)
        return mean, logvar
        
        
class SmilesDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, pad_idx, num_layers, dropout, device=None):
        super(SmilesDecoder, self).__init__()
        ## params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cpu') if device is None else device
        
        ## special tokens
        self.pad_idx = pad_idx
        
        ## Neural Network
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size, padding_idx=self.pad_idx)
        self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.dense = nn.Sequential(nn.Linear(self.hidden_size + self.latent_size, self.hidden_size), nn.ReLU())
        self.output2vocab = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, inp, z, hidden=None):
        '''
        Params
        ------
        inp.shape = (batch, 1)
        z.shape = (batch, latent)
        hidden.shape = (numlayer, batch, hidden)
        '''
        if hidden is None:
            return self._forward(inp, z)
        else:
            return self._forward_single(inp, hidden, z)
        
        
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device) # hidden.shape = (numlayer, batch, hidden)
        return hidden
        
    
    def _forward(self, inp, z):
        '''
        Params
        ------
        inp.shape = (batch, seq)
        z.shape = (batch, latent)
        '''
        batch_size = inp.size(0)
        seqlen = inp.size(1)
        
        ## Embedding
        inp_emb = self.embedding(inp) # inp_emb.shape = (batch, seq, hidden)
        
        ## Condition
        cond = z.unsqueeze(1) # cond.shape = (batch, 1, latent)
        cond = cond.repeat(1,seqlen,1) # cond.shape = (batch, seq, latent)
        inp_cond = torch.cat((inp_emb, cond), 2) # inp_cond.shape = (batch, seq, hidden + latent)
        inp_cond = self.dense(inp_cond) # inp_cond.shape = (batch, seq, hidden)
        
        ## Decoder - Teacher forcing
        out, _ = self.rnn(inp_cond) # out.shape = (batch, seq, hidden)
        
        ## Prediction
        logits = self.output2vocab(out) # logits.shape = (batch, seq, vocab)
        return logits
        
    
    def _forward_single(self, inp, hidden, z):
        '''
        Params
        ------
        inp.shape = (batch, 1)
        hidden.shape = (numlayer, batch, hidden)
        z.shape = (batch, latent)
        '''
        batch_size = inp.size(0)
        
        ## Embedding
        inp_emb = self.embedding(inp) # inp_emb.shape = (batch, 1, hidden)
        
        ## Condition
        inp = torch.cat((inp_emb, z.unsqueeze(1)), 2) # inp.shape = (batch, 1, hidden + latent)
        inp = self.dense(inp) # inp.shape = (batch, 1, hidden)
        
        ## Decoder - Teacher forcing
        out, hidden = self.rnn(inp, hidden) # out.shape = (batch, 1, hidden), hidden.shape = (numlayer, batch, hidden)
        
        ## Prediction
        logits = self.output2vocab(out) # logits.shape = (batch, 1, vocab)
        return logits, hidden
        

class SmilesAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, sos_idx, eos_idx, pad_idx, num_layers=2, dropout=0., device=None, filepath_config=None):
        super(SmilesAutoencoder, self).__init__()
        
        ## params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cpu') if device is None else device
        self.filepath_config = filepath_config

        ## special tokens
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        ## is there predefined configurations?
        if self.filepath_config is not None:
            self.load_config(self.filepath_config)
            
        ## model build
        self.encoder = SmilesEncoder(self.vocab_size, self.hidden_size, self.latent_size, self.pad_idx, self.num_layers, self.dropout, self.device)
        self.decoder = SmilesDecoder(self.vocab_size, self.hidden_size, self.latent_size, self.pad_idx, self.num_layers, self.dropout, self.device)

        ## device
        self.to(self.device)

    
    def molecular_transform(self, dataset, K=20, use_tqdm=True):
        ## Flag of GPU
        use_cuda = torch.cuda.is_available()
        
        ## tqdm
        loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=use_cuda)
        if use_tqdm:
            loader = tqdm.tqdm(loader, total=len(dataset))
        
        ## Transform one-by-one
        generated = []
        for batch in loader:
            batch_smiles = dataset.encode(batch["smiles_s"], batch["length_s"].max())
            batch_length = batch["length_s"]
            ## predict
            for k in range(K):
                seq = self.predict(batch_smiles, batch_length)
                smi = dataset.decode(seq)[0] # note: batch_size = 1
                generated.append((batch["smiles_s"][0][1:-1], smi))   
        df_generated = pd.DataFrame.from_records(generated)
        return df_generated
    
    
    def policy_gradient(self, dataset, reward_ft,
                        batch_size=1000, total_steps=2000, learning_rate=1e-4, discount_factor=0.995, buffer_size=2000, buffer_batch_size=50,
                        validation_dataset=None, validation_repetition_size=20,
                        display_step=10, validation_step=100, checkpoint_step=100, checkpoint_filepath=None, verbose=1):
        ## Flag of GPU
        use_cuda = torch.cuda.is_available()
        
        ## Initialization for training
        self.history = []
        self.history_valid = []
        replay_buffer = ReplayBufferDataset()
        
        start_time = time.time()
        for step in range(1,total_steps+1):
            ## Generate episodes to fill a replay buffer
            for batch in DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, pin_memory=use_cuda):
                ## input data preprocess
                single_encode = dataset.encode(batch["smiles_s"], batch["length_s"].max()) # single_encode.shape = (1, seq)
                single_length = batch["length_s"] # single_length.shape = (1, )
                
                ## mini-batch for replay buffer
                batch_inp_encode = single_encode.repeat(buffer_batch_size, 1) # batch_inp_encode.shape = (buffer_batch, seq)
                batch_inp_length = single_length.repeat(buffer_batch_size) # batch_inp_length.shape = (buffer_batch, )

                ## generate target data
                batch_out_encode = self.predict(batch_inp_encode, batch_inp_length) # batch_out_encode.shape = (sample, max_seqlen)
                batch_tar_smiles = [dataset.sos_char + smi + dataset.eos_char for smi in dataset.decode(batch_out_encode)] # len(batch_tar_smiles) = batch
                
                ## reward
                smi_src = batch["smiles_s"][0]
                batch_properties = np.array([reward_ft(smi_src[1:-1], smi_tar[1:-1]) for smi_tar in batch_tar_smiles], dtype=self._get_default_dtype())
                
                ## append
                for smi_tar, (rew_tar, sim_tar, pro_tar) in zip(batch_tar_smiles, batch_properties):
                    if rew_tar > 0:
                        replay_buffer.push(smi_src, smi_tar, rew_tar, sim_tar, pro_tar)
                        
                ## replay pretraining data
                smi_tar = batch["smiles_t"][0]
                batch_properties = np.array([reward_ft(smi_src[1:-1], smi_tar[1:-1])], dtype=self._get_default_dtype())
                rew_tar = batch_properties[0,0]
                sim_tar = batch_properties[0,1]
                pro_tar = batch_properties[0,2]
                replay_buffer.push(smi_src, smi_tar, rew_tar, sim_tar, pro_tar)
                
                ## is buffer full?
                if len(replay_buffer) >= buffer_size:
                    break
                    
            ## Buffer statistics
            avg_reward, avg_similarity, avg_properties = replay_buffer.stats()
            
            ## Replay episodes for training
            for batch in DataLoader(replay_buffer, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False):
                batch_encode_src = dataset.encode(batch["smiles_src"], batch["length_src"].max())
                batch_length_src = batch["length_src"]
                batch_encode_tar = dataset.encode(batch["smiles_tar"], batch["length_tar"].max())
                batch_length_tar = batch["length_tar"]
                batch_reward = batch["reward"].to(self.device)
                ## policy gradient
                rl_loss = self.partial_policy_gradient(batch_encode_src, batch_length_src,
                                                       batch_encode_tar, batch_length_tar, batch_reward,
                                                       lr=learning_rate, gamma=discount_factor)
                ## buffer update
                replay_buffer.pop()
                break
            
            ## history
            self.history.append((rl_loss, avg_reward, avg_similarity, avg_properties))
            
            ## model save
            if (checkpoint_filepath is not None) and (step % checkpoint_step == 0):
                self.save_model(checkpoint_filepath)
            
            ## log
            if step % display_step == 0:
                log = f"[{step:06d}/{total_steps:06d}]"
                log += f"  loss: {rl_loss:.3f}"
                log += f"  reward: {avg_reward:.3f}"
                log += f"  similarity: {avg_similarity:.3f}"
                log += f"  property: {avg_properties:.3f}"
                
                if validation_dataset is not None and (step % validation_step == 0):
                    df_generated_valid = self.molecular_transform(validation_dataset, K=validation_repetition_size, use_tqdm=False)
                    properties_valid = []
                    for smi_src, smi_tar in df_generated_valid.values:
                        outs_val = reward_ft(smi_src, smi_tar)
                        sim_val = outs_val[1]
                        prop_val = outs_val[2]
                        properties_valid.append((smi_src, smi_tar, sim_val, prop_val))

                    df_metrics_valid = evaluate_metric_validation(pd.DataFrame.from_records(properties_valid), validation_repetition_size)
                    df_metrics_valid = df_metrics_valid.T.rename(index={0:step})
                    self.history_valid.append(df_metrics_valid)
                    log += f"  valid_ratio(va): {df_metrics_valid.loc[step, 'VALID_RATIO']:.3f}"
                    log += f"  similarity(va): {df_metrics_valid.loc[step, 'AVERAGE_SIMILARITY']:.3f}"
                    log += f"  property(va): {df_metrics_valid.loc[step, 'AVERAGE_PROPERTY']:.3f}"
                
                end_time = time.time()
                log += f"  ({(end_time - start_time) / 60:.1f} min)"
                print(log)
            
        df_history_valid = pd.concat(self.history_valid)
        df_history = pd.DataFrame(self.history, columns=["LOSS", "REWARD", "SIMILARITY", "PROPERTY"])
        return df_history, df_history_valid
    
    
    def partial_policy_gradient(self, smiles_A, length_A, smiles_B, length_B, rewards, lr=1e-4, gamma=0.995):
        '''
        Params
        ------
        smiles_A.shape = (batch, seq)
        length_A.shape = (batch, )
        smiles_B.shape = (batch, seq)
        length_B.shape = (batch, )
        rewards.shape = (batch, )
        '''
        batch_size = smiles_B.size(0)
        seqlen = smiles_B.size(1)
        
        ## Training phase
        self.encoder.eval()
        self.decoder.train()
        
        ## Optimizer
        optim_decoder = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        optim_decoder.zero_grad()
        
        ## Encoder
        with torch.no_grad():
            mean_A, logvar_A = self.encoder(smiles_A, length_A) # mean_A.shape = (batch, latent), logvar_A.shape = (batch, latent)
        
        ## Sampling
        z_A = self.encoder.sampling(mean_A, logvar_A) # z_A.shape = (batch, latent)
        
        ## Decode
        logits_B = self.decoder(smiles_B, z_A) # logits_B.shape = (batch, seq, vocab)
        logp_B = torch.nn.functional.log_softmax(logits_B, dim=-1) # logp.shape = (batch, seq, vocab)
        
        ## Returns (= cummulative rewards = discounted rewards)
        G_B = torch.zeros(batch_size, seqlen, device=self.device) # G_B.shape = (batch, seq)
        G_B[torch.arange(batch_size), length_B-1] = rewards
        for t in range(1, seqlen):
            G_B[:,-t-1] = G_B[:,-t-1] + G_B[:,-t] * gamma
        
        ## Loss
        glogp_B = G_B.unsqueeze(-1) * logp_B # glogp_B.shape = (batch, seq, vocab)
        target_ravel = smiles_B[:,1:].contiguous().view(-1) # target_ravel.shape = (batch*(seq-1), )
        glogp_ravel = glogp_B[:,:-1,:].contiguous().view(-1, glogp_B.size(-1)) # logp_ravel.shape = (batch*(seq-1), vocab)
        rl_loss = nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(glogp_ravel, target_ravel)
        
        ## Backpropagation
        rl_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.) # gradient clipping
        optim_decoder.step()
        
        return rl_loss.item()
    
    
    def fit(self, dataset,
            batch_size=100, total_steps=100000, learning_rate=1e-3,
            validation_dataset=None, validation_repetition_size=20,
            display_step=1000, checkpoint_step=10000, checkpoint_filepath=None,
            margin=1., verbose=1):
        ## Flag of GPU
        use_cuda = torch.cuda.is_available()
        
        ## Initialization of optimizer
        optim_encoder = torch.optim.AdamW(self.encoder.parameters(), lr=learning_rate)
        optim_decoder = torch.optim.AdamW(self.decoder.parameters(), lr=learning_rate)
        
        ## Initialization for training
        step = 0
        history = []
        history_valid = []
        
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
                batch_loss = self.partial_fit(*batch_inputs, optim_encoder=optim_encoder, optim_decoder=optim_decoder, m=margin)
                
                ## loss for training data
                loss_train         = batch_loss[0]
                loss_recon_A_train = batch_loss[1] # L(src)
                loss_recon_B_train = batch_loss[2] # L(tar)
                loss_recon_C_train = batch_loss[3] # L(neg)
                loss_kld_train     = batch_loss[4] # L(src,tar)
                loss_triplet_train = batch_loss[5] # L(src,neg) + L(tar,neg)
                
                ## history
                history.append((loss_train,
                                loss_recon_A_train,
                                loss_recon_B_train,
                                loss_recon_C_train,
                                loss_kld_train,
                                loss_triplet_train))
                
                ## model save
                if (checkpoint_filepath is not None) and (step % checkpoint_step == 0):
                    self.save_model(checkpoint_filepath)
                
                ## log
                if (verbose > 0) and (step % display_step == 0):
                    log = f"[{step:08d}/{total_steps:08d}]"
                    log += f"  loss(tr): {loss_train:.3f}"
                    log += f"  loss_recon_src(tr): {loss_recon_A_train:.3f}"
                    log += f"  loss_recon_tar(tr): {loss_recon_B_train:.3f}"
                    log += f"  loss_recon_neg(tr): {loss_recon_C_train:.3f}"
                    log += f"  loss_kld(tr): {loss_kld_train:.3f}"
                    log += f"  loss_triplet(tr): {loss_triplet_train:.3f}"
                    
                    if (validation_dataset is not None) and (step % checkpoint_step == 0):
                        df_generated_valid = self.molecular_transform(validation_dataset, K=validation_repetition_size, use_tqdm=False)
                        properties_valid = []
                        for smi_src, smi_tar in df_generated_valid.values:
                            sim_val = similarity(smi_src, smi_tar)
                            prop_val = 0.999 # this value is not important and just for logging
                            properties_valid.append((smi_src, smi_tar, sim_val, prop_val))
                        
                        df_metrics_valid = evaluate_metric_validation(pd.DataFrame.from_records(properties_valid), validation_repetition_size)
                        df_metrics_valid = df_metrics_valid.T.rename(index={0:step})
                        history_valid.append(df_metrics_valid)
                        log += f"  valid_ratio(va): {df_metrics_valid.loc[step, 'VALID_RATIO']:.3f}"
                        log += f"  similarity(va): {df_metrics_valid.loc[step, 'AVERAGE_SIMILARITY']:.3f}"
                        
                    print(log)
                
                ## termination
                if step >= total_steps: break
                step += 1
                
        df_history_valid = pd.concat(history_valid)
        df_history = pd.DataFrame(history, columns=["LOSS_TOTAL",
                                                    "LOSS_RECONSTRUCTION_SOURCE",
                                                    "LOSS_RECONSTRUCTION_TARGET",
                                                    "LOSS_RECONSTRUCTION_NEGATIVE",
                                                    "LOSS_KL_DIVERGENCE",
                                                    "LOSS_TRIPLET"])
                              
        return df_history, df_history_valid
    
    
    def partial_fit(self, smiles_A, length_A, smiles_B, length_B, smiles_C, length_C, optim_encoder, optim_decoder, m):
        '''
        Params
        ------
        smiles.shape = (batch, seq)
        length.shape = (batch, )
        '''
        batch_size = smiles_A.size(0)
        assert batch_size == smiles_B.size(0)
        
        ## Training phase
        self.train()
        
        ## Optimizer
        optim_encoder.zero_grad()
        optim_decoder.zero_grad()
        
        ## Encoder
        mean_A, logvar_A = self.encoder(smiles_A, length_A) # mean_A.shape = (batch, latent), logvar_A.shape = (batch, latent)
        mean_B, logvar_B = self.encoder(smiles_B, length_B)
        mean_C, logvar_C = self.encoder(smiles_C, length_C)

        ## Sampling
        z_A = self.encoder.sampling(mean_A, logvar_A) # z_A.shape = (batch, latent)
        z_B = self.encoder.sampling(mean_B, logvar_B)
        z_C = self.encoder.sampling(mean_C, logvar_C)

        ## Decoder
        logits_A = self.decoder(smiles_A, z_A) # logits_A.shape = (batch, seq, vocab)
        logits_B = self.decoder(smiles_B, z_B) # logits_B.shape = (batch, seq, vocab)
        logits_C = self.decoder(smiles_C, z_C) # logits_C.shape = (batch, seq, vocab)

        ## Reconstruction loss
        loss_recon_A = self._calc_reconstruction_loss(logits_A, smiles_A)
        loss_recon_B = self._calc_reconstruction_loss(logits_B, smiles_B)
        loss_recon_C = self._calc_reconstruction_loss(logits_C, smiles_C)
        loss_recon = 0.3 * loss_recon_A + 0.3 * loss_recon_B + 0.4 * loss_recon_C
        
        ## KL divergence
        loss_kld_A = self._calc_kl_divergence_loss(mean_A, logvar_A)
        loss_kld_B = self._calc_kl_divergence_loss(mean_B, logvar_B)
        loss_kld_C = self._calc_kl_divergence_loss(mean_C, logvar_C)
        loss_kld = 0.3 * loss_kld_A + 0.3 * loss_kld_B + 0.4 * loss_kld_C
        
        ## Triplet loss
        '''
        Ishfaq, Haque, A. Hoogi, and D. Rubin. "TVAE: Deep metric learning approach for variational autoencoder."
        Workshop at International Conference on Learning Representations (ICLRW). Vol. 32. 2018.
        '''
        loss_triplet = self._calc_triplet_loss(mean_A, mean_B, mean_C, m)

        ## Total loss
        loss = loss_kld + 0.5 * loss_recon + 10. * loss_triplet
        
        ## Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.) # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.) # gradient clipping
        optim_encoder.step()
        optim_decoder.step()
        
        return loss.item(), loss_recon_A.item(), loss_recon_B.item(), loss_recon_C.item(), loss_kld.item(), loss_triplet.item()


    def _calc_reconstruction_loss(self, logit, target):
        '''
        Params
        ------
        target.shape = (batch, seq)
        logit.shape = (batch, seq, vocab)
        '''
        logp = torch.nn.functional.log_softmax(logit, dim=-1) # logp.shape = (batch, seq, vocab)
        target_ravel = target[:,1:].contiguous().view(-1) # target_ravel.shape = (batch*(seq-1), )
        logp_ravel = logp[:,:-1,:].contiguous().view(-1, logp.size(2)) # logp_ravel.shape = (batch*(seq-1), vocab)
        loss = nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(logp_ravel, target_ravel)
        return loss


    def _calc_kl_divergence_loss(self, mean, logvar):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0)
        return kld_loss

    
    def _calc_triplet_loss(self, mean_A, mean_B, mean_C, m):
        d_AB = self._calc_euclidean_distance(mean_A, mean_B)
        d_AC = self._calc_euclidean_distance(mean_A, mean_C)
        triplet_loss = torch.mean(torch.nn.functional.relu(d_AB - d_AC + m), dim=0)
        return triplet_loss
        
        
    def _calc_euclidean_distance(self, mean_A, mean_B):
        return torch.sqrt(torch.sum(torch.pow(mean_A - mean_B, 2), dim=1))
    
    
    def transform(self, smiles, length):
        mean, logvar = self.encoder(smiles, length)
        return mean.cpu().detach().numpy(), logvar.cpu().detach().numpy()
    
    
    def predict(self, smiles, length, max_seqlen=128):
        '''
        Params
        ------
        smiles.shape = (batch, seq)
        length.shape = (batch, )
        '''
        ## evaluate phase
        self.eval()
        ## Params
        batch_size = smiles.size(0)
        ## Generation
        with torch.no_grad():
            ## Encoder
            mean, logvar = self.encoder(smiles, length) # mean.shape = (batch, latent), logvar.shape = (batch, latent)
            ## Sampling
            z = self.encoder.sampling(mean, logvar) # z.shape = (batch, latent)
            ## Decoder
            generated = []
            for i in range(batch_size):
                z_i = z[i].unsqueeze(0) # z_i.shape = (1, latent)
                seq = self._generate(z_i, max_seqlen) # seq.shape = (1, max_seqlen)
                seq = seq.cpu().numpy()
                generated.append(seq)
        generated = np.concatenate(generated) # generated.shape = (batch, max_seqlen)
        return generated
            
            
    def _generate(self, z, max_seqlen, greedy=False):
        '''
        Params
        ------
        z.shape = (1, latent)
        '''
        batch_size = z.size(0)
        
        ## Initialize outs
        outs = torch.full(size=(batch_size, max_seqlen), fill_value=self.pad_idx, dtype=torch.long, device=self.device) # outs.shape = (batch, max_seqlen)
        
        ## Initial hidden
        hiddens = self.decoder.init_hidden(batch_size) # hiddens.shape = (numlayer, batch, hidden)
        
        ## Start token
        inps_sos = torch.full(size=(batch_size, 1), fill_value=self.sos_idx, dtype=torch.long, device=self.device) # inps_sos.shape = (batch, 1)
        
        ## Recursive
        inps = inps_sos # inps.shape = (batch, 1)
        for i in range(max_seqlen):
            ## Terminal condition
            if inps[0][0] == self.eos_idx or inps[0][0] == self.pad_idx:
                outs[:,i] = self.eos_idx
                break
            else:
                outs[:,i] = inps
                
            ## Decode
            logits, hiddens = self.decoder(inps, z, hiddens) # logits.shape = (batch, 1, vocab), hiddens.shape = (numlayer, batch, hidden)

            ## Next word
            if greedy:
                _, top_idx = torch.topk(logits, 1, dim=-1) # top_idx.shape = (batch, 1, 1)
                inps = top_idx.contiguous().view(batch_size, 1) # inps.shape = (batch, 1)
            else:
                probs = torch.softmax(logits, dim=-1) # probs.shape = (batch, 1, vocab)
                probs = probs.view(probs.size(0), probs.size(2)) # probs.shape = (batch, vocab)
                inps = torch.multinomial(probs, 1) # inps.shape = (batch, 1)
        return outs


    def _get_default_dtype(self):
        ddtype_torch = torch.get_default_dtype()
        if ddtype_torch is torch.float32: # is equivalent to 'torch.float'
            return np.float32
        elif ddtype_torch is torch.float64: # is equivalent to 'torch.double'
            return np.float64
            

    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    
    def save_config(self, path):
        with open(path, 'w') as fout:
            fout.write(f"VOCAB_SIZE,{self.vocab_size}\n")
            fout.write(f"HIDDEN_SIZE,{self.hidden_size}\n")
            fout.write(f"LATENT_SIZE,{self.latent_size}\n")
            fout.write(f"NUM_LAYERS,{self.num_layers}\n")
            fout.write(f"DROPOUT,{self.dropout}\n")
            fout.write(f"SOS_IDX,{self.sos_idx}\n")
            fout.write(f"EOS_IDX,{self.eos_idx}\n")
            fout.write(f"PAD_IDX,{self.pad_idx}\n")
            
            
    def load_config(self, path):
        with open(path) as fin:
            lines = fin.readlines()
        lines = [l.rstrip().split(",") for l in lines]
        ## parsing
        params = dict()
        for k, v in lines:
            params[k] = v
        ## update configs
        self.vocab_size = int(params["VOCAB_SIZE"])
        self.hidden_size = int(params["HIDDEN_SIZE"])
        self.latent_size = int(params["LATENT_SIZE"])
        self.num_layers = int(params["NUM_LAYERS"])
        self.dropout = float(params["DROPOUT"])
        self.sos_idx = int(params["SOS_IDX"])
        self.eos_idx = int(params["EOS_IDX"])
        self.pad_idx = int(params["PAD_IDX"])
        