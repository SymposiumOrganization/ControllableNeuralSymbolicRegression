import numpy as np
import multiprocessing
import click
import warnings
from tqdm import tqdm
from pathlib import Path
import pickle
import time

from ControllableNesymres.dataset import generator
from ControllableNesymres import dclasses
from ControllableNesymres.utils import (
    create_env,
    H5FilesCreator,
    return_number_of_equations,
    code_unpickler,
    code_pickler,
)
import copyreg
import types
import traceback
import h5py



def handle_resume_and_cleanup(folder_path, resume, eq_per_block, number_of_blocks):
    if not (resume and folder_path.exists()):
        print("Folder does not exist or resume is False. Starting from scratch")
        print("Total number of h5 to generate:", number_of_blocks)
        print("Each h5 contains", eq_per_block, "equations")
        print("Total number of equations to generate:", number_of_blocks * eq_per_block)
        return 0, eq_per_block, False

    files = sorted(int(x.stem) for x in folder_path.glob("*.h5") if x.stem.isdigit())

    if not files:
        raise ValueError(f"Folder {folder_path} empy or it does not contain any hdf, beside metadata.h5. Please delete the folder and start again")

    old_name_to_new_name = cleanup_hdf_files(files, folder_path)

    starting_block = len(old_name_to_new_name)
    eqs_per_block_old = return_number_of_equations(folder_path / f"{files[0]}.h5")
    
    print("h5 generated so far", len(old_name_to_new_name), "out of", number_of_blocks)
    print("Each h5 contains", eqs_per_block_old, "equations")
    print("Total number of equations generated so far:", len(old_name_to_new_name) * eqs_per_block_old)

    return starting_block, eqs_per_block_old, True


def cleanup_hdf_files(files, folder_path):
    old_name_to_new_name = {x: i for i, x in enumerate(files)}
    for old_name in files:
        new_name = old_name_to_new_name[old_name]
        old_path = folder_path / f"{old_name}.h5"
        new_path = folder_path / f"{new_name}.h5"
        old_path.rename(new_path)

    return old_name_to_new_name


@click.command()
@click.option(
    "--number_of_equations",
    default=200,
    help="Total number of equations to generate",
)
@click.option(
    "--eq_per_block",
    default=10000,
    help="Total number of equations for each hd5 file",
)
@click.option(
    "--cores",
    default=1,
    help="Run in debug mode (1 process) or in parallel. By default is 1 process",
)
@click.option(
    "--root_folder_path",
    default="data",
    help="Path to the folder where the data will be stored",
)
@click.option("--resume", is_flag=True, help="Resume the creation of the dataset")
def creator(
    number_of_equations, eq_per_block, cores, root_folder_path, resume):

    # Register pickling functions for multiprocessing
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
    
    cpus_available = multiprocessing.cpu_count()

    if cores > cpus_available:
        print(f"Warning: you are using more cores than available. Using all the available cores {cpus_available}, instead")
        cores = cpus_available

    eq_per_block = min(number_of_equations // cores, eq_per_block)
    print(
        "There are {} equations per block. The progress bar will have this resolution".format(
            eq_per_block
        )
    )
    warnings.filterwarnings("error")
    env, param, config_dict = create_env("configs/dataset_configuration.json")

    folder_path = Path(root_folder_path) / f"raw_datasets/{number_of_equations}"
    number_of_blocks = np.ceil(number_of_equations / eq_per_block).astype(int)
    starting_block, eq_per_block, resume = handle_resume_and_cleanup(folder_path, resume, eq_per_block, number_of_blocks)
    
    h5_creator = H5FilesCreator(target_path=folder_path)

    env_pip = generator.Pipepile(env, 
                      number_of_equations=number_of_equations, 
                      eq_per_block=eq_per_block,
                      h5_creator=h5_creator,
                      is_timer=True,
                      )
    start_time = time.time()
    if cores > 1:
        try:
            with multiprocessing.Pool(cores) as p:
                with tqdm(total=number_of_blocks) as pbar:
                    for f in p.imap_unordered(
                        env_pip.create_block, range(starting_block, number_of_blocks)
                    ):
                        pbar.update()
        except:
            print(traceback.format_exc())


    else:
        list(map(env_pip.create_block, tqdm(range(starting_block, number_of_blocks))))
    
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
                            total_number_of_eqs=number_of_equations,
                            eqs_per_hdf=eq_per_block,
                            generator_details=param)
    
    print("Expression generation took {} seconds".format(time.time() - start_time))
    folder_path.mkdir(parents=True, exist_ok=True)
    t_hf = h5py.File(folder_path / "metadata.h5" , 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(dataset)))
    t_hf.close()


if __name__ == "__main__":
    creator()