import click
import numpy as np
from ControllableNesymres.utils import load_eq, load_metadata_hdf5
from ControllableNesymres.dataset.data_utils import evaluate_fun
from ControllableNesymres.dataset.data_utils import create_uniform_support, evaluate_fun, return_dict_metadata_dummy_constant
from torch.distributions.uniform import Uniform
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path


class Pipeline:
    def __init__(self, data_path, metadata, support):
        self.data_path = data_path
        self.metadata = metadata
        self.support = support

    def is_valid_and_not_in_validation_set(self, idx:int) -> bool:
        """
        Return True if an equation is not the validation set and is numerically meaningfull (i.e. values all differs from nan, -inf, +inf, all zeros),
               We test whether is in the validation dataset both symbolically and numerically
        Args:
            idx: index of the equation in the dataset
        
        """
        try:
            eq = load_eq(self.data_path, idx, self.metadata.eqs_per_hdf)
        except FileNotFoundError:
            return idx, False
        dict_costs = return_dict_metadata_dummy_constant(self.metadata)
        consts = torch.stack([torch.ones([int(self.support.shape[1])])*dict_costs[key] for key in dict_costs.keys()])
        input_lambdi = torch.cat([self.support,consts],axis=0)
        assert input_lambdi.shape[0]  == len(self.metadata.total_coefficients) + len(self.metadata.total_variables)

        #Numerical Checking        
        args = [ eq.code,input_lambdi ]
        y = evaluate_fun(args)

        curr = [x if not np.isnan(x) else "nan" for x in y] 
        val = tuple(curr)
        if val == tuple([]):
            # Not an equation
            return idx, False
        if val == tuple([float("-inf")]*input_lambdi.shape[-1]):
            # All Inf
            return idx, False
        if val == tuple([float("+inf")]*input_lambdi.shape[-1]):
            # All +Inf
            return idx, False
        if val == tuple([float(0)]*input_lambdi.shape[-1]):
            # All zeros
            return idx, False
        if val == tuple(["nan"]*input_lambdi.shape[-1]):
            # All nan
            return idx, False
        return idx, True

        

@click.command()
@click.option("--data_path", default="data/raw_datasets/10000000/", help="Path to the dataset created with create_dataset.py")
@click.option("--debug/--no-debug", default=False)
def main(data_path,debug):
    print("Loading metadata")
    data_path = Path(data_path)
    metatada = load_metadata_hdf5(data_path)
    sampling_distribution = Uniform(-25,25) 
    num_p = 400
    support = create_uniform_support(sampling_distribution, len(metatada.total_variables), num_p)
    print("Creating image for validation set")
    pipe = Pipeline(data_path, metatada, support)
    print("Starting finding out index of equations present in the validation set or wih numerical problems")
    total_eq = int(metatada.total_number_of_eqs)
    res = []
    if not debug:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            with tqdm(total=total_eq) as pbar:
                for evaled in p.imap_unordered(pipe.is_valid_and_not_in_validation_set, list(range(total_eq)),chunksize=10000):
                    pbar.update()
                    res.append(evaled)
    else:
        res = list(map(pipe.is_valid_and_not_in_validation_set, tqdm(range(total_eq))))
    
    print("Total number of equations processed", len(res))
    total = len(res)
    assert total == len(set([x[0] for x in res]))
    good_equations = len([x for x in res if x[1]])
    bad_equations = len([x for x in res if not x[1]])
    print(f"Total number of good equations {good_equations} over {total}, {good_equations/total*100}%")
    print(f"Total number of bad equations {bad_equations} over {total}, {bad_equations/total*100}%")

    np.save(data_path/ "equations_validity.npy",res)

if __name__=="__main__":
    main()