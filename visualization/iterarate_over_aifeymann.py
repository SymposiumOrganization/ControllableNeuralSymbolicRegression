
from nesymres.architectures.data import description2tokens
from nesymres.utils import load_metadata_hdf5
import hydra
from pathlib import Path
from functools import partial
import torch
from sympy import lambdify
import json
import omegaconf
from nesymres.utils import return_fitfunc
from nesymres.dataset.data_utils import timeout_return_symmetry, extract_complexity
from nesymres.architectures.data import constants_to_placeholder,create_subtrees_info,wrap_symmetry_info, set_noise_of_equation, prepare_examples, add_conditional_entries_to_word2id, decompose_equation,create_complexity_info
from nesymres.architectures import data
from nesymres.dataset.generator import Generator
from sympy import simplify, sympify
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from collections import Counter

load_dotenv(find_dotenv())


def get_info(info_eq,eq,cfg,metadata,noise):
    info_eq = {}
    ## complexity ##
    info_eq=create_complexity_info(info_eq,eq)
    ## positives ##
    eq_sympy_infix = data.constants_to_placeholder(eq)
    eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
    with open(hydra.utils.to_absolute_path(cfg.path_to_candidate)) as f:
        eq_candidates = json.load(f)
    eq_candidates = [eq.replace(" ", "") for eq in eq_candidates]
    eq_candidates = sorted(eq_candidates, key=len, reverse=False)
    eqs_candidate = (list(metadata.word2id.keys()) + eq_candidates, set(list(metadata.word2id) + eq_candidates)) # Idea order them in length and then sample from the shortest ones
    info_eq =create_subtrees_info(info_eq, eq_sympy_prefix, eqs_candidate, metadata=metadata, cfg=cfg)

    # # temporarily kill negative examples
    # info_eq["negative_prefix_examples"]=[]
    # info_eq['negative_examples']=[]


    info_eq["noise_level"] = noise



    
    # Pad the info_eq
    #info_eq["tokenized"] =  info_eq["tokenized"] + [0 for x  in range(15)]
    return info_eq


def get_info_new(info_eq,eq,cfg,metadata,noise):
    info_eq = {}
    ## complexity ##
    info_eq=create_complexity_info(info_eq,eq)
    ## positives ##
    eq_sympy_infix = constants_to_placeholder(eq)
    eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
    with open(hydra.utils.to_absolute_path(cfg.path_to_candidate)) as f:
        eq_candidates = json.load(f)
    eq_candidates = [eq.replace(" ", "") for eq in eq_candidates]
    eq_candidates = sorted(eq_candidates, key=len, reverse=False)
    eqs_candidate = (list(metadata.word2id.keys()) + eq_candidates, set(list(metadata.word2id) + eq_candidates)) # Idea order them in length and then sample from the shortest ones
    info_eq =create_subtrees_info(info_eq, eq_sympy_prefix, eqs_candidate, metadata=metadata,cfg=cfg)
    # temporarily kill negative examples
    info_eq["negative_prefix_examples"]=[]
    info_eq['negative_examples']=[]
    ## noise ##
    info_eq["noise_level"] = list(str(noise))
    ## symmetry ##
    info_eq['symmetry']= timeout_return_symmetry(eq,metadata.config['variables'],n_support=3)
    info_eq = wrap_symmetry_info(info_eq,0)
    return info_eq




def main():
    df = pd.read_csv("test_set/aifeymann_processed.csv")
    # Iterate over all rows

    for index, row in df.iterrows():
        ## Load metadata.h5 file
        cfg = omegaconf.OmegaConf.load(os.getenv("CONFIG_PATH"))
        cfg.inference.bfgs.activated=False
        cfg.inference.beam_size=10
        cfg.inference.bfgs.add_coefficients_if_not_existing=False
        weights_path = os.getenv("WEIGHTS")
        eq_setting = load_metadata_hdf5(Path(os.getenv("DATASET_PATH")))
        fitfunc = return_fitfunc(cfg, eq_setting, weights_path)

        # Create points from an equation
        number_of_points = 500
        n_variables = 2

        #To get best results make sure that your support inside the max and mix support
        max_supp = 3#cfg.dataset_train.fun_support["max"]
        min_supp = 1# cfg.dataset_train.fun_support["min"]
        # Torch fix random seed
        torch.manual_seed(0)
        import numpy as np
        np.random.seed(0)

        X = torch.rand(number_of_points,len(eq_setting.total_variables))*(max_supp-min_supp)+min_supp 
        X = X #+ torch.rand(number_of_points,len(eq_setting.total_variables))*0.5
        X[:,n_variables:] = 0
        target_eq = row["eq"] #Use x_1,x_2 and x_3 as independent variables)
        print()
        print("Target equation: ",target_eq)
        print("Real complexity: ", extract_complexity(target_eq))
        X_dict = {x:X[:,idx].cpu() for idx, x in enumerate(eq_setting.total_variables)} 


        y = lambdify(",".join(eq_setting.total_variables), target_eq)(**X_dict)
        

        noise_entry = ["epsilon=0"]
        noise = 0
        y = y  + torch.normal(0,noise,size=y.shape)

        print("X shape: ", X.shape)
        print("y shape: ", y.shape)
        #cond = torch.ones(1,25,device="cuda")*2
        # cond = torch.ones(1,25)*2
        # cond[0,1] = 1

        symbols = ["x_1","x_2","x_3","x_4","x_5"]
        info_eq = {}
        

        if cfg.conditioning.mode == "transf":
            components = list()    
            eq_sympy_infix = data.constants_to_placeholder(target_eq)
            eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
            decompose_equation(eq_sympy_prefix, components=components, metadata=eq_setting)
            positive_prefix_examples = set(components)
            
            positive_prefix_examples = []#["sin(x_2)"]
            negative_prefix__examples = []
            positive_examples, negative_examples = prepare_examples(positive_prefix_examples, negative_prefix__examples,  eq_setting.word2id, info=True)
            info_eq["positive_examples"] = positive_examples
            info_eq["negative_examples"] = negative_examples
            print(extract_complexity(simplify(target_eq)))
            


            with open(hydra.utils.to_absolute_path('data/conditioning/equations_ops_3_5000.json')) as f:
                eq_candidates = json.load(f)

            eq_candidates = [eq.replace(" ", "") for eq in eq_candidates]
            
            # Sort by length
            metadata=eq_setting
            eq_candidates = sorted(eq_candidates, key=len, reverse=False)
            eqs_candidate = (list(metadata.word2id.keys()) + eq_candidates, set(list(metadata.word2id) + eq_candidates)) # Idea order them in length and then sample from the shortest ones
            probability_of_sampling = [1/len(eq) for eq in eqs_candidate[0]] 




            info_eq = {}
            cfg.predict_c=False
            cfg.constants=False
            cfg.dataset_train.predict_c=False
            info_eq = get_info_new(info_eq,target_eq,cfg,metadata,noise)
            #info_eq["positive_prefix_examples"] = [['add', 'x_3']]
            info_eq["symmetry"]= ["<mask>" for x in range(25)]
            info_eq["positive_prefix_examples"] = [["mul","sqrt","2"]]
            info_eq["negative_prefix_examples"] = []
            info_eq = create_complexity_info(info_eq, eq_sympy_infix, masking_prob=0.)
            info_eq["noise_level"] = ["epsilon=0"]
            info_eq["tokenized"] = description2tokens(info_eq, metadata.word2id, cfg)
            cond = torch.tensor(info_eq["tokenized"]).unsqueeze(0)


        else:
            cond = None

        output = fitfunc(X,y,cond)

        
        new_output = []
        for i,x in enumerate(output["all_bfgs_preds"]):        
            x = sympify(x)
            new_output.append(x)
            print(i,x)
            print(extract_complexity(x))
        

            # Sympyfy the expression


        

if __name__ == "__main__":
    main()