import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .set_encoder import SetEncoder #, SymEncoder
from .beam_hypothesis import BeamHypotheses
import numpy as np
from ..dataset.generator import Generator, InvalidPrefixExpression
from sympy import lambdify 
from . import bfgs
from . import sym_encoder
from . data import  de_tokenize, replace_ptr_with_costants, extract_variables_from_infix
from ..dclasses import BFGSParams
import json
import pandas as pd
import sympy as sp
import math
from pathlib import Path
import random
import pickle
import hydra
from joblib import Parallel, delayed
import timeout_decorator
from .utils import stable_r2_score
from sympy import  parse_expr
import threading
from collections import defaultdict

bfgs_timeout = timeout_decorator.timeout(700)(bfgs.bfgs)

class Model(pl.LightningModule):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.enc = SetEncoder(cfg.architecture)

        if cfg.dataset.conditioning.mode == True and cfg.architecture.conditioning == False:
            print("Warning: conditioning is set to True in the dataset config file but to False in the architecture config file. Conditioning will ignored")
        
        if cfg.architecture.conditioning != False:
            # The following assert only checks for the training dataloader.
            assert cfg.dataset.conditioning.mode == True, "Inconsistency in Conditioning options:  architecture.conditioning is True but dataset.conditioning.mode is False. Please make sure that both are set to True or False"
        
 
        if cfg.architecture.conditioning == True or cfg.architecture.conditioning == "v3": # v3 was the old name for the current conditioning mode
            self.symenc = sym_encoder.SymEncoderWithAttentionDec(cfg)   
        elif cfg.architecture.conditioning == False:
            self.symenc = None
        else:
            raise KeyError("Conditioning mode not recognized")
        
        self.trg_pad_idx = cfg.architecture.trg_pad_idx
        self.tok_embedding = nn.Embedding(cfg.architecture.number_possible_tokens, cfg.architecture.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.architecture.length_eq, cfg.architecture.dim_hidden)
        if cfg.architecture.sinuisodal_embeddings:
            self.create_sinusoidal_embeddings(
                cfg.architecture.length_eq, cfg.architecture.dim_hidden, out=self.pos_embedding.weight
            )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.architecture.dim_hidden,
            nhead=cfg.architecture.num_heads,
            dim_feedforward=cfg.architecture.dec_pf_dim,
            dropout=cfg.architecture.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.architecture.dec_layers)
        self.fc_out = nn.Linear(cfg.architecture.dim_hidden, cfg.architecture.number_possible_tokens)
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dropout = nn.Dropout(cfg.architecture.dropout)
        self.eq = None

        self.bfgs = BFGSParams(
                activated= cfg.inference.bfgs.activated,
                n_restarts=cfg.inference.bfgs.n_restarts,
                add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
                normalization_o=cfg.inference.bfgs.normalization_o,
                idx_remove=cfg.inference.bfgs.idx_remove,
                normalization_type=cfg.inference.bfgs.normalization_type,
                stop_time=cfg.inference.bfgs.stop_time,
            )
        self.cnt = 0
        self.cnt_ep=0
        self.metadata = None # See train
        self.mapper = None # See train 
        self.target_expr = set() 


   

    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_src_mask_enc(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask


    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask

    def forward(self,batch):
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, : (size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)
        trg = batch[1].long()
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])
        src_mask = None
        encoder_input = torch.cat((src_x, src_y), dim=-1)

        enc_src = self.enc(encoder_input)
        if self.cfg.architecture.conditioning == True or self.cfg.architecture.conditioning == "v3":
            babba = self.symenc(batch[3], device=self.device)
            enc_src = enc_src + babba

        elif self.cfg.architecture.conditioning == False:
            pass
        else:
            raise KeyError("Conditioning not implemented. Available v3 or False")

        assert not torch.isnan(enc_src).any()
        pos = self.pos_embedding(
            torch.arange(0, batch[1].shape[1] - 1)
            .unsqueeze(0)
            .repeat(batch[1].shape[0], 1)
            .type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        trg_ = self.dropout(te + pos)
        output = self.decoder_transfomer(trg_.permute(1, 0, 2),enc_src.permute(1, 0, 2),trg_mask2.bool(),tgt_key_padding_mask=trg_mask1.bool())#,memory_key_padding_mask=mask_dec.bool()) 
        output = self.fc_out(output)

        return output, trg

    def compute_loss(self,output, trg):
        output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return loss

    def compute_loss_per_sample(self, output, trg):
        output = output.permute(1, 0, 2)
        loss_total = []
        for i in range(output.shape[0]):
            curr_trg = trg[i, 1:]
            loss = self.criterion(output[i], curr_trg)
            # If loss is nan
            # if torch.isnan(loss):
            #     breakpoint()
            x = loss.item()
            # print(x)
            loss_total.append(x)
        return loss_total
    #[x[1]["cost_to_pointer"] for x in batch[2]]
    def training_step(self, batch, _):
        if self.cnt_ep > 4 and self.cfg.resume_from_checkpoint:
            raise MemoryError("Memory error")
        if batch[0].shape[0] == None:
            return None
        output, trg = self.forward(batch)

        # Save equation randomly
        if random.randint(0,10000) > 9999:
            str_eq = [str(x) for x in  batch[2]]
            
            if Path("eqs.json").exists():
                with open("eqs.json","r") as f:
                    json_content = json.load(f)

                json_content.extend(str_eq)
            else:
                json_content = str_eq

            with open(f"eqs_{self.cnt}.json","w") as f:
                json.dump(str_eq, f)
            
            self.cnt = self.cnt + 1
            # if 
            # with open("eqs","r") as f:
        
        target_expr = set([tuple(x[1]["target_expr"]) for x in batch[2]])
        self.target_expr.update(target_expr)
        self.cnt = self.cnt + 1
        loss = self.compute_loss(output,trg)
        self.log("train_loss", loss, on_epoch=True, batch_size=batch[0].shape[0])
        return loss
    
    



    def validation_step(self,batch, batch_idx, dataloader_idx):

        if batch[0].shape[0] == 0:
            return 0

        output, trg = self.forward(batch)
        loss = self.compute_loss(output,trg)
        dataset_name = self.mapper[dataloader_idx]
        self.log(f"val_loss_{dataset_name}", loss, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        
        
        # Perform beam search for each element in the batch without enabling constants
        X = batch[0][:,:-1, :].permute(0, 2, 1)
        y = batch[0][:,-1, :]
        
        # # From {"numerical_cond": torch.tensor, "symbolic_cond": torch.tensor} to [{"numerical_cond": 1Dtensor, "symbolical_cond": 1Dtensor}, ...]
        if self.cfg.dataset.conditioning.mode == True:
            cpu_numpy_batch_3 = {key: value.cpu().numpy() for key, value in batch[3].items() }
            cpu_numpy_batch_3 = [{key: value[idx] for key, value in cpu_numpy_batch_3.items()} for idx in range(len(batch[2]))]
            cond_str = [entry[1] for entry in batch[2]]
        else:
            cpu_numpy_batch_3 = [None for _ in range(len(batch[2]))]
            cond_str = [None for _ in range(len(batch[2]))]

        # Create a dictionary with all information
        if self.trainer.global_rank:
            self.cnt_ep=self.cnt_ep+5
        to_return = {
            "gt": [batch[2][idx][0] for idx in range(len(batch[2]))],
            "cond_str": cond_str,
            "cond_raw": cpu_numpy_batch_3,
            
            #"output": output,
            "X": X.cpu().numpy(),
            "y": y.cpu().numpy(),
        }
        # to_return['cond_str'][10]['condition_str_tokenized']
        # 
        # breakpoint()
        # Save the dictionary in a folder in res folder

        if self.cfg.resume_from_checkpoint != "":
            root_path = Path(hydra.utils.to_absolute_path(self.cfg.resume_from_checkpoint)).parent
            folder_name = root_path / Path(f"res/{self.current_epoch}")
        else:
            folder_name = Path(f"res/{self.current_epoch}")
        folder_name.mkdir(parents=True, exist_ok=True)
        file_name = folder_name / f"{dataset_name}_{batch_idx}_{self.trainer.global_rank}.pkl"
        path = file_name
        with open(path, "wb") as f:
            pickle.dump(to_return, f)

        # Save the set files
        if self.target_expr:
            file_name = folder_name / f"seen_eqs_{self.trainer.global_rank}.pkl"
            path = file_name
            with open(path, "wb") as f:
                pickle.dump(self.target_expr, f)
        # else:
        #     to_return = None

        self.target_expr = set()

        return None


    def validation_epoch_end(self, outputs):
        """
        Ignore what is going on in validation_step, we open a new csv
        """

        self.cnt_ep=self.cnt_ep+5
        # breakpoint()               
        return None # Ignored for now
 


    def configure_optimizers(self):
    # set LR to 1, scale with LambdaLR scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=1, weight_decay=0.01)

        # constant warmup, then 1/sqrt(n) decay starting from the initial LR
        lr_func = lambda step: min(self.cfg.architecture.lr, self.cfg.architecture.lr / math.sqrt(max(step, 1)/self.cfg.architecture.wupsteps))
        #self.log("train_loss", loss, on_epoch=True, batch_size=batch[0].shape[0])

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
        }]


    def fitfunc(self, X,y,cond, cond_str=None, is_batch=False, val_X=None, val_y=None, cfg_params=None):
        """Same API as fit functions in sklearn if is_batch if False
            X [Number_of_points, Number_of_features], 
            Y [Number_of_points]
            If is_batch is True, X and Y are lists of the same length
            X [Number_of_batches, Number_of_points, Number_of_features],
            Y [Number_of_batches, Number_of_points]
            cond_str used for rejection sampling
            
        """

        if not is_batch:
            X = X
            y = y[:,None]

            if type(X) != torch.Tensor:
                X = torch.tensor(X,device=self.device).unsqueeze(0)
            else:
                X = X.unsqueeze(0)
            if X.shape[2] < self.cfg.architecture.dim_input - 1:
                pad = torch.zeros(1, X.shape[1],self.cfg.architecture.dim_input-X.shape[2]-1, device=self.device)
                X = torch.cat((X,pad),dim=2)
            if type(y) != torch.Tensor:
                y = torch.tensor(y,device=self.device).unsqueeze(0)
            else:
                y = y.unsqueeze(0)
            
            if type(cond) == dict:
                for key, value in cond.items():
                    if type(value) != torch.Tensor:
                        cond[key] = torch.tensor(value,device=self.device).unsqueeze(0)
                    else:
                        cond[key] = value.unsqueeze(0)

            elif type(cond) != torch.Tensor:
                cond = torch.tensor(cond,device=self.device).unsqueeze(0)
            else:
                cond = cond.unsqueeze(0)
                
            bs = 1
            assert X.shape[0] == 1
            assert y.shape[0] == 1
            
        elif is_batch:
            bs = len(X)
            y = y.unsqueeze(2)

        n_words = self.cfg.architecture.number_possible_tokens
        assert len(X.shape) == 3
        assert len(y.shape) == 3
        assert X.shape[0] == y.shape[0]
        assert y.shape[2] == 1
        encoder_input = torch.cat((X, y), dim=2) #.permute(0, 2, 1)
        src_enc = self.enc(encoder_input)
       
        if self.cfg.architecture.conditioning == True or self.cfg.architecture.conditioning == "v3":
            babba = self.symenc(cond, device=self.device)
            src_enc = src_enc+babba 
        elif self.cfg.architecture.conditioning == False:
            print("No conditioning")
        else:
            raise ValueError("Conditioning not implemented")
        
        max_length = self.cfg.architecture.length_eq 

        bbs = (bs, cfg_params.beam_size,) + src_enc.shape[1:]
        bxb = bs * cfg_params.beam_size
        bxbs = ((bxb,) + src_enc.shape[1:])
        
        src_enc = src_enc.unsqueeze(1).expand(bbs).contiguous().view(bxbs)
        src_len = torch.ones((bs, cfg_params.beam_size), device=self.device).long() * max_length
        
        assert src_enc.size(0) == bxb
        generated = torch.zeros([bxb, max_length], dtype=torch.long, device=self.device)
        generated[:, 0] = 1
        
        # generated = torch.tensor(trg_indexes,device=self.device,dtype=torch.long)
        generated_hyps = [BeamHypotheses(cfg_params.beam_size, max_length, 1.0, 1) for _ in range(bs)]
        
        # positions # Not sure what this is for
        positions = src_len.new(max_length).long()
        positions = torch.arange(max_length, out=positions).unsqueeze(0).expand_as(generated)
        
        
        # Beam Scores
        beam_scores = torch.zeros((bs, cfg_params.beam_size), device=self.device, dtype=torch.long)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
  
        # cache compute states
        cache = {"slen": 0}
        
        # done sentences
        done = [False for _ in range(bs)]
        
        #print("Before autoregressive loop")
        while cur_len < max_length:
            #print("cur_len- max_lengt", cur_len, max_length)

            # breakpoint()
            generated_mask1, generated_mask2 = self.make_trg_mask(
                generated[:, :cur_len]
            )
            
            pos = self.pos_embedding(
                torch.arange(0, cur_len)  #### attention here
                .unsqueeze(0)
                .repeat(generated.shape[0], 1)
                .type_as(generated)
            )
            te = self.tok_embedding(generated[:, :cur_len])
            trg_ = self.dropout(te + pos)

            output = self.decoder_transfomer(
                trg_.permute(1, 0, 2),
                src_enc.permute(1, 0, 2),
                generated_mask2.float(),
                tgt_key_padding_mask=generated_mask1.bool(),
            )
            output = self.fc_out(output)
            output = output.permute(1, 0, 2).contiguous()
            
            assert output[:, -1:, :].shape == (bxb,1,self.cfg.architecture.number_possible_tokens,)
            
            tensor = output[:, -1, :] # (bs * beam_size, n_words)
            scores = F.log_softmax(tensor, dim=-1)#.squeeze(1)
            assert scores.size() == (bxb, n_words)

            n_words = scores.shape[-1]
            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, cfg_params.beam_size * n_words)  # (bs, beam_size * n_words)

            if True:
                next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=1, largest=True, sorted=True)

                assert next_scores.size() == next_words.size() == (bs, 2 * cfg_params.beam_size)
                
                next_batch_beam = []
                # if bs>1:
                #     breakpoint()
                
                # for each sentence
                for sent_id in range(bs):
                    
                    done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                    
                    if done[sent_id]:
                        next_batch_beam.extend([(0, cfg_params.word2id["P"], 0)] * cfg_params.beam_size)  # pad the batch
                        continue
                    
                    next_sent_beam = []
                        
                    # next words for this sentence
                    
                    for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                        # get beam and word IDs
                        beam_id =  torch.div(idx, n_words, rounding_mode='floor')
                        word_id = idx % n_words

                        # end of sentence, or next word
                        if (
                            word_id == cfg_params.word2id["F"]
                            or cur_len + 1 == self.cfg.architecture.length_eq
                        ):
                            generated_hyps[sent_id].add(generated[sent_id * cfg_params.beam_size + beam_id,:cur_len].clone().cpu(),value.item())
                        else:
                            next_sent_beam.append((value, word_id, sent_id * cfg_params.beam_size +  beam_id))

                        # the beam for next step is full
                        if len(next_sent_beam) == cfg_params.beam_size:
                            break

                    # update next beam content
                    assert (len(next_sent_beam) == 0 if cur_len + 1 == self.cfg.architecture.length_eq else cfg_params.beam_size)
                    if len(next_sent_beam) == 0:
                        next_sent_beam = [
                            (0, self.trg_pad_idx, 0)
                        ] * cfg_params.beam_size  # pad the batch


                    next_batch_beam.extend(next_sent_beam)
                    assert len(next_batch_beam) == cfg_params.beam_size * (sent_id + 1)

                
                assert len(next_batch_beam) == bxb
                beam_scores = torch.tensor(
                    [x[0] for x in next_batch_beam], device=self.device
                )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                beam_words = torch.tensor(
                    [x[1] for x in next_batch_beam], device=self.device
                )  # generated.new([x[1] for x in next_batch_beam])
                beam_idx = torch.tensor(
                    [x[2] for x in next_batch_beam], device=self.device
                )
            
                generated = generated[beam_idx, :]
                generated[:,cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

                # update current length
                cur_len = cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

        
        cfg_params.id2word[3] = "constant"

        print("After autoregressive loop")

        # Format for Parallel
        conds = []
        for i in range(X.shape[0]):
            conds.append({
                "symbolic_conditioning":cond['symbolic_conditioning'][i],
                "numerical_conditioning":cond['numerical_conditioning'][i]
                })
            
        X = X.cpu()
        y = y.cpu()
        cond_numbers = []
        for i, _ in enumerate(conds):
            curr = {}
            curr["symbolic_conditioning"] = conds[i]["symbolic_conditioning"].cpu( )
            curr["numerical_conditioning"] = conds[i]["numerical_conditioning"].cpu()
            cond_numbers.append(curr)
        
        if cond_str is None or cond_str == []:
            cond_str = [None for _ in range(X.shape[0])]

        # We can used parallel processing here
        if X.shape[0] == 1:
            # Create a new generated_hyps where we have only one hypothesis for each entry            
            new_generate_hyps = [BeamHypotheses(n_hyp=1, max_len=60, length_penalty=None, early_stopping=None) for _ in range(cfg_params.beam_size)]
            for idx, curr in enumerate(generated_hyps[0].hyp):
                prob, tens = curr

                new_generate_hyps[idx].add(tens, score=prob)

            if val_X is None:
                val_X = None 
                val_y = None
            else:
                val_X = val_X[0]
                val_y = val_y[0]

            eqs = Parallel(n_jobs=cfg_params.n_jobs)(
                delayed(from_hyps_to_expr)(
                    hypotheses, X[:1].clone(),y[:1].clone(), cond_numbers[0], cond_str[0], val_X=val_X, val_y=val_y, cfg_params=cfg_params,
                )
                for i, hypotheses in enumerate(new_generate_hyps)
            )
            # Find the index with the lowest loss
            target_metric = cfg_params.target_metric
            all_metrics = [curr["best_loss"][target_metric] for curr in eqs]

            if target_metric in ["r2","r2_val"]:
                # Replace nans with -inf
                all_metrics = np.array(all_metrics)
                all_metrics[np.isnan(all_metrics)] = -np.inf

                # Sort in descending order
                idx_order = np.argsort(all_metrics)[::-1]
            elif target_metric in ["mse"]:
                # Replace nans with inf
                all_metrics = np.array(all_metrics)
                all_metrics[np.isnan(all_metrics)] = np.inf
                
                idx_order = np.argsort(all_metrics)                
        
            if target_metric in ["r2","r2_val"]:
                # Check if we have just nans
                if np.all(np.isnan(all_metrics)):
                    idx = 0
                else:
                    idx = np.nanargmax(all_metrics)
            elif target_metric in ["mse"]:
                idx = np.nanargmin(all_metrics)
            else:
                raise KeyError("Unknown target metric")

            
            # Reshape eqs. From a list of dictionaries to a dictionary of lists
            keys_available = eqs[0].keys() 
            eqs_reshaped = defaultdict(list)
            eqs_reshaped["all_losses"] = {}
            for i in range(len(eqs)):
                for key in keys_available:
                    if key == "best_loss":
                        entry = eqs[i][key]
                        for key2 in entry.keys():
                            if key2 in eqs_reshaped["all_losses"]:
                                eqs_reshaped["all_losses"][key2].append(entry[key2])
                            else:
                                eqs_reshaped["all_losses"][key2] = [entry[key2]]
                    elif key == "all_losses":
                        for key2 in eqs[i][key][0].keys():
                            if eqs[i]["all_losses"][0][key2] is not None and  ~np.isnan(eqs[i]["all_losses"][0][key2]) and abs(eqs[i]["all_losses"][0][key2]) != np.inf:
                                assert eqs[i]["best_loss"][key2] == eqs[i]["all_losses"][0][key2]
                        
                    elif isinstance(eqs[i][key], list) or isinstance(eqs[i][key], np.ndarray) and eqs[i][key].shape[0] == 1:
                        if eqs[i][key]:
                            entry = eqs[i][key][0]
                        else:
                            entry = None
                        eqs_reshaped[key].append(entry)
                    else:
                        entry = eqs[i][key]
                        eqs_reshaped[key].append(entry)
            


            eqs_reshaped["sorted_idx"] = idx_order
            eqs_reshaped["best_pred"] = eqs_reshaped["all_preds"][idx]
            eqs_reshaped["best_loss"] = {key: value[idx] for key,value in eqs_reshaped["all_losses"].items()}
            eqs = eqs_reshaped

        else:
            if val_X is None:
                val_X = [None for _ in range(X.shape[0])]

            if val_y is None:
                val_y = [None for _ in range(X.shape[0])]
            eqs = Parallel(n_jobs=cfg_params.n_jobs)(
                delayed(from_hyps_to_expr)(
                    hypotheses, X[i:i+1].clone(),y[i:i+1].clone(), cond_numbers[i], cond_str[i], val_X=val_X[i:i+1], val_y=val_y[i:i+1], cfg_params=cfg_params,
                )
                for i, hypotheses in enumerate(generated_hyps)
            )

        return eqs

    def get_equation(self,):
        return self.eq

#@timeout(120)
def from_hyps_to_expr(hypotheses_i, X_i,y_i, cond_i, cond_str_i, val_X=None, val_y=None, cfg_params=None):
    losses = []
    preds_bfgs = []
    preds_raw = []            
    for __, ww in sorted(
        hypotheses_i.hyp, key=lambda x: x[0], reverse=True
    ):  
        
        prefix_with_ptr = de_tokenize(ww[1:].tolist(), cfg_params.id2word)
        #bfgs_loss = None
        try:
            infix_with_ptr = Generator.prefix_to_infix(prefix_with_ptr, coefficients=["constant"], variables=cfg_params.total_variables)
        except InvalidPrefixExpression:
            print("Cannot prefix to infix" + str(prefix_with_ptr))
            losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
            preds_bfgs.append('illegal parsing infix')
            preds_raw.append('illegal parsing infix')
            continue
        print("Expr: ", infix_with_ptr)
        try:
            prefix_with_c = replace_ptr_with_costants(prefix_with_ptr, cond_i['numerical_conditioning'])
        except ValueError:
            print("Cannot replace constants" + str(prefix_with_ptr))
            losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
            preds_bfgs.append('illegal pointers')
            preds_raw.append('illegal pointers')

            #preds_bfgs.append(str(prefix_with_ptr))
            #preds_raw.append(infix_with_ptr)
            continue
        #expression = get_expression(ww, cfg_params)
        infix_with_c = Generator.prefix_to_infix(prefix_with_c, coefficients=["constant"], variables=cfg_params.total_variables)
        infix_with_c = infix_with_c.format(constant="constant")
        symbols = {i: sp.Symbol(f'c{i}') for i in range(infix_with_c.count("constant"))} 

        if cfg_params.evaluate == False:
            try:
                infix_with_c = str(parse_expr(infix_with_c))
                preds_raw.append(infix_with_ptr)
                preds_bfgs.append(str(infix_with_c))
            except:
                print("Cannot parse" + str(prefix_with_ptr))
                losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
                preds_bfgs.append('illegal parse')
                preds_raw.append('illegal parse')
            continue
        
        if cfg_params.bfgs.activated and len(symbols) > 0:
            try:
                if threading.current_thread() is threading.main_thread():
                    pred, _, loss, _ = bfgs_timeout(infix_with_c, X_i, y_i, cfg=cfg_params)
                else:
                    pred, _, loss, _ = bfgs.bfgs(infix_with_c, X_i, y_i, cfg=cfg_params)
                    


            # assert threading.current_thread() is threading.main_thread()
            # try:
            #     pred, _, loss, _ = bfgs.bfgs(infix_with_c, X_i, y_i, cfg=cfg_params)
            except (timeout_decorator.timeout_decorator.TimeoutError, ZeroDivisionError, IndexError, NameError,TypeError, KeyError,TypeError,OverflowError,ZeroDivisionError):
                losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
                print("Bfgs failed with" + str(infix_with_c))
                # Print the stack trace of the exception
                import traceback
                traceback.print_exc()
            
                preds_bfgs.append('illegal bfgs')
                preds_raw.append('illegal bfgs')
                continue
            
            bfgs_loss = loss
            
            if np.isnan(loss):
                losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})      
                preds_bfgs.append(str(pred))
                preds_raw.append(infix_with_ptr)
                continue

            infix = str(pred)

        elif not cfg_params.bfgs.activated and len(symbols) > 0:
            losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
            preds_raw.append(infix_with_ptr)
            preds_bfgs.append(str(infix_with_c))
            continue
        
        elif cfg_params.bfgs.activated and len(symbols) ==  0:
            infix = infix_with_c

        else:
            raise ValueError("This should not happen")

        y_curr = y_i.squeeze()
        X_curr = X_i.clone().half()
        try:
            infix = str(parse_expr(infix))
            vars_list = extract_variables_from_infix(infix)
            indeces = [int(x[2:])-1 for x in vars_list]
        except (SyntaxError, AttributeError,ValueError):
            print("Canniot parse expression" + str(prefix_with_ptr))
            losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
            preds_bfgs.append("illegal parse")
            preds_raw.append("illegal parse")
            continue

        
        
        X_curr = X_curr[:,:,indeces]
       
        try:
            f = lambdify(vars_list, infix,  modules=["numpy",{'asin': np.arcsin, "ln": np.log, "Abs": np.abs}])
            y_pred = f(*X_curr.squeeze(0).T)
            if val_X is not None and val_X[0] is not None:
                y_val_pred = f(*val_X.squeeze(0).T)
            else:
                y_val_pred = None

        except (NameError,IndexError, RuntimeError, KeyError,TypeError,OverflowError,ZeroDivisionError,timeout_decorator.timeout_decorator.TimeoutError):
            losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
            preds_bfgs.append(str(prefix_with_ptr))
            preds_raw.append(infix_with_ptr)
            continue
        
        
        loss = np.mean(np.square(y_pred-y_curr).numpy())

        # if val_data:
        #     breakpoint()
        # if not bfgs_loss is None:

        #     assert np.isclose(loss, bfgs_loss)
        
        # Compute r2 score
        try:
            r2 = stable_r2_score(y_curr, y_pred)
            if y_val_pred is not None:
                r2_val = stable_r2_score(val_y.squeeze(), y_val_pred)
            else:   
                r2_val = None
        except:
            losses.append({'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan})
            preds_bfgs.append(str(prefix_with_ptr))
            preds_raw.append(infix_with_ptr)
            continue
        

        losses.append({'mse':loss, 'r2':r2, 'r2_val':r2_val})
        preds_raw.append(infix_with_ptr)
        preds_bfgs.append(str(infix))

       
        
    #if cfg_params.bfgs.activated:       
    metric_chosen = cfg_params.target_metric    
    candidates = np.array([x[metric_chosen] for x in losses])

    if all(np.isnan(np.array(candidates))):
        print("Warning all nans")
        best_raw_pred = None
        best_pred = None
        best_loss = {'mse':np.nan, 'r2':np.nan, 'r2_val':np.nan}
        idx = -1
        ordered_idx = []
    else:
        if metric_chosen in ["r2","r2_val"]:
            # We want to maximize r2
            #best_loss = np.nanmax(candidates)
            idx = np.nanargmax(candidates)

            # Sort the idx by r2 from best to worst
            ordered_idx = np.argsort(candidates)[::-1]

        elif metric_chosen in ["mse"]:
            #best_loss = np.nanmin(candidates)
            idx = np.nanargmin(candidates)
            ordered_idx = np.argsort(candidates)
        else:
            raise ValueError("Unknown metric")

        best_loss = losses[idx]
        best_pred = preds_bfgs[idx]
        best_raw_pred = preds_raw[idx]
    

    output = {'all_raw_preds':preds_raw, 'all_preds':preds_bfgs, 'all_losses':losses, 'best_raw_pred':best_raw_pred, 'best_pred':best_pred, 'best_loss':best_loss, "idx_equation":idx, "sorted_idx":ordered_idx}
    return output

