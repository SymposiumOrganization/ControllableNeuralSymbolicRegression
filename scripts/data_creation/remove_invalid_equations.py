import numpy as np
import multiprocessing
import click
import os
from ControllableNesymres.utils import load_metadata_hdf5
from pathlib import Path
from ControllableNesymres.utils import H5FilesCreator
from tqdm import tqdm
import h5py
import pickle

def create_hdf_files(metadata, keep_true, base_path: Path, target_path: Path, debug) -> None:
    num_eqs_per_set = metadata.eqs_per_hdf

    n_datasets = int(len(keep_true) // num_eqs_per_set) + 1
    h5_creator = H5FilesCreator(base_path, target_path,metadata)
    sets = [[keep_true[idx] for idx in range(i*num_eqs_per_set,min((i+1)*num_eqs_per_set,len(keep_true)))] for i in range(n_datasets)]
    if not debug:
        available_cpu = multiprocessing.cpu_count()
        # In order to avoid memory issues, we use only a fraction of the available cpu
        available_cpu = min(int(available_cpu * 0.25),1)
        with multiprocessing.Pool(available_cpu) as p: #multiprocessing.cpu_count()) as p:
            max_ = n_datasets
            with tqdm(total=max_) as pbar:
                for f in p.imap_unordered(
                    h5_creator.recreate_single_hd5_from_idx, enumerate(sets)
                ):
                    pbar.update()
    else:
        t = map(h5_creator.recreate_single_hd5_from_idx, tqdm(enumerate(sets)))
    total_number_of_eqs = len(keep_true)
    metadata.eqs = []
    metadata.total_number_of_eqs = total_number_of_eqs

    assert metadata.eqs_per_hdf == num_eqs_per_set
    t_hf = h5py.File(target_path/ "metadata.h5" , 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(metadata)))
    t_hf.close()
    return


@click.command()
@click.option("--data_path", default="data/raw_datasets/10000000/")
@click.option("--debug/--no-debug", default=False)
def main(data_path,debug):
    data_path = Path(data_path)
    bool_cond = np.load(data_path / "equations_validity.npy" ,allow_pickle=True)
    entries = [idx for idx, entry in bool_cond if entry]
    metatada = load_metadata_hdf5(data_path)
    target_path = Path(data_path.parent.parent / Path("datasets") / data_path.stem)
    if target_path.exists():
        raise ValueError("Target path already exists, delete it first")
    create_hdf_files(metatada, entries, data_path, target_path, debug)
    




if __name__=="__main__":
    main()