import marshal
import copyreg
import types
import pickle
import json
from .dataset import generator
from .dclasses import DatasetDetails, Equation, GeneratorDetails
from typing import List, Tuple
import h5py
import os
import numpy as np
from pathlib import Path
from ControllableNesymres.dclasses import FitParams, BFGSParams
from functools import partial
import ControllableNesymres
import sys
sys.modules['nesymres'] = ControllableNesymres 

class H5FilesCreator():
    def __init__(self,base_path: Path = None, target_path: Path = None, metadata=None):
        target_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.target_path = target_path
        
        self.base_path = base_path
        self.metadata = metadata
        

    def create_single_hd5_from_eqs(self,block):
        name_file, eqs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5") , 'w')
        for i, eq in enumerate(eqs):            
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()
    
    def recreate_single_hd5_from_idx(self,block:Tuple):
        name_file, eq_idxs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5") , 'w')
        for i, eq_idx in enumerate(eq_idxs):            
            eq = load_eq_raw(self.base_path, eq_idx, self.metadata.eqs_per_hdf)
            #curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=eq)
        t_hf.close()


def code_unpickler(data):
    return marshal.loads(data)

def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)

def load_eq_raw(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx/num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder,f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file)*int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    f.close()
    return raw_metadata

def return_number_of_equations(path_folder):
    f = h5py.File(path_folder, 'r')
    eqs = len(f.keys())
    f.close()
    return eqs


def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx/num_eqs_per_set))
    try:
        f = h5py.File(os.path.join(path_folder,f"{index_file}.h5"), 'r')
    except FileNotFoundError as e:
        print("Issue with file: ", os.path.join(path_folder,f"{index_file}.h5"))
        raise KeyError("Issue with file: ", os.path.join(path_folder,f"{index_file}.h5"))
    except OSError as e:
        print("Issue with file: ", os.path.join(path_folder,f"{index_file}.h5"))
        raise KeyError("Issue with file: ", os.path.join(path_folder,f"{index_file}.h5"))
    dataset_metadata = f[str(idx - int(index_file)*int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    f.close()
    return metadata

def load_metadata_hdf5(path_folder: Path) -> DatasetDetails:
    f = h5py.File(path_folder / "metadata.h5", 'r')
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata

def create_env(path)->Tuple[generator.Generator,GeneratorDetails]:
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    env = generator.Generator(param)
    return env, param, d

def retrofit_word2id(metadata, cfg):
    # Retrofit word2id if there is conditioning
    if cfg.architecture.conditioning != False:
        from ControllableNesymres.architectures.data import add_conditional_entries_to_word2id, word_creator
        complexity_words, symmetry_words = word_creator(metadata, cfg)
        metadata.word2id = add_conditional_entries_to_word2id( metadata.word2id, complexity_words, symmetry_words)
        previous = metadata.id2word
        metadata.id2word = {v: k for k,v in metadata.word2id.items()}

        # Make sure that all the words appear in previous are also in the new id2word with the same id
        for k,v in previous.items():
            assert metadata.id2word[k] == v

    else:
        print("Conditioning is not activated")
    return metadata

def return_fitfunc(cfg, metadata, weights_path, device='cpu'):
    ## Set up BFGS load rom the hydra config yaml
    bfgs = BFGSParams(
            activated= cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

    # Retrofit word2id if there is conditioning
    metadata = retrofit_word2id(metadata, cfg)
    # elif cfg.architecture.conditioning == False:
    #     pass
    # else:
    #     raise KeyError("Conditioning must be either True or False")
    if not "evaluate" in cfg.inference:
        cfg.inference.evaluate = True
    if not "rejection_sampling" in cfg.inference:
        cfg.inference.rejection_sampling = False
        cfg.inference.rejection_sampling_n_samples = 0
    if not "target_metric" in cfg.inference:
        cfg.inference.target_metric = "mse"
    
    params_fit = FitParams(word2id= metadata.word2id, 
                            id2word={int(k): v for k,v in metadata.id2word.items()}, 
                            una_ops=metadata.una_ops, 
                            bin_ops=metadata.bin_ops, 
                            total_variables=metadata.total_variables,  
                            total_coefficients=metadata.total_coefficients,
                            rewrite_functions=metadata.rewrite_functions,
                            bfgs=bfgs,
                            beam_size=cfg.inference.beam_size, #This parameter is a tradeoff between accuracy and fitting time
                            n_jobs=cfg.inference.n_jobs,
                            evaluate=cfg.inference.evaluate,
                            rejection_sampling=cfg.inference.rejection_sampling,
                            rejection_sampling_n_samples=cfg.inference.rejection_sampling_n_samples,
                            metadata=metadata,
                            target_metric=cfg.inference.target_metric,
                            )
    ## Load architecture, set into eval mode, and pass the config parameters
    from ControllableNesymres.architectures.model import Model

    if device=='cpu':
        model = Model.load_from_checkpoint(weights_path, cfg=cfg)
    else:
        model = Model.load_from_checkpoint(weights_path, cfg=cfg)
        model= model.cuda()
    model = model.eval()

    fitfunc = partial(model.fitfunc,cfg_params=params_fit)
    return fitfunc


