import pandas as pd 
import numpy as np
from multiprocessing import Manager
import click
import warnings
from tqdm import tqdm
import json
import os
from ControllableNesymres.dataset import generator
import time
import signal
from ControllableNesymres import dclasses
from pathlib import Path
import pickle
from sympy import lambdify, sympify
from ControllableNesymres.utils import create_env, H5FilesCreator
from ControllableNesymres.utils import code_unpickler, code_pickler
import copyreg
import types
from itertools import chain
import traceback
import sympy as sp
import h5py

    


@click.command()
@click.option("--folder_csv", default="test_set/icml2023")
def converter(folder_csv):
    """
    This scripts iterate over all datasets in iclr2023 and create a validation dataloader for each dataset.
    """
    csv_availables = Path(folder_csv).glob("*.csv")
    for file_csv in csv_availables:
        validation = pd.read_csv(file_csv)
        copyreg.pickle(types.CodeType, code_pickler, code_unpickler) #Needed for serializing code objects
        env, param, config_dict = create_env("configs/dataset_configuration.json")
        
        dataloader_name = Path(file_csv).stem 
        folder_path = Path("data/benchmark") / dataloader_name
        folder_path.mkdir(parents=True, exist_ok=True)
        h5_creator = H5FilesCreator(target_path=folder_path)
        number_of_equations = len(validation)
        eqs_per_block = min(number_of_equations,5000)
        env_pip = generator.Pipepile(env, number_of_equations, eqs_per_block, h5_creator, is_timer=False)
        res = []
        for idx in range(len(validation)):
            gt_expr = validation.iloc[idx]["eq"]
            gt_expr = gt_expr.replace("pow","Pow")
            gt_expr = gt_expr.replace("I","1")
            gt_expr = gt_expr.replace("sqrt(2)","1.4142135623730951" )
            
            variables = list(eval(validation.iloc[idx]["support"]).keys())
            support = validation.iloc[idx]["support"]
            
            curr = env_pip.convert_lambda(gt_expr,variables,support) 
            #expr_without_constants = sp_expr_to_skeleton(sympify(curr.expr))
            #print("Without constants: ", expr_without_constants)
            #curr.expr = expr_without_constants
            res.append(curr)

            print("Converting: ", gt_expr)
            print("Converted: ", curr.expr)
            print()

        print("Finishing generating set")
        h5_creator.create_single_hd5_from_eqs(("0", res))
        dataset = dclasses.DatasetDetails(
                                config=config_dict, 
                                total_coefficients=env.coefficients, 
                                total_variables=list(env.variables), 
                                word2id=env.word2id, 
                                id2word=env.id2word,
                                una_ops=env.una_ops,
                                bin_ops=env.bin_ops,
                                operators=env.operators,
                                rewrite_functions=env.rewrite_functions,
                                total_number_of_eqs=len(res),
                                eqs_per_hdf=len(res),
                                generator_details=param)

        t_hf = h5py.File(os.path.join(folder_path, "metadata.h5") , 'w')
        t_hf.create_dataset("other", data=np.void(pickle.dumps(dataset)))
        t_hf.close()
    


if __name__ == "__main__":
    converter()