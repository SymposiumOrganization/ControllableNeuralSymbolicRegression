import pickle
from pathlib import Path
import omegaconf
from ControllableNesymres.utils import return_fitfunc
from ControllableNesymres.utils import load_metadata_hdf5
from ControllableNesymres.architectures.utils import compute_accuracy_pointwise
from ControllableNesymres.architectures.data import tokenize, create_positives_and_constants, mix_ptr_constants, compute_properties, match_properties, get_robust_random_data, extract_variables_from_infix
from tqdm import tqdm
import torch 
import time 
from joblib import Parallel, delayed
import sympy as sp
from ControllableNesymres.dataset.generator import Generator

def return_entries(current, root_file_name, file, original_number_of_point=None, eq_setting=None, cfg=None, do_sympy_check=False, experiment_type=None):
    number_of_point = current["X"].shape[0]
    if current["gt"] == 0:
        return None
    
    if current["outputs"]["best_pred"] is None:
        print(f"Warning: The best_pred is None for equation {str(current['gt'])}")
        
        if len(current["outputs"]["all_raw_preds"]):
            print("Using the first prediction instead.")
            tmp = current["outputs"]["all_raw_preds"][0]
            current["outputs"]["best_pred"] = tmp
        else:
            print("No predictions available. Setting it to 0.")
            current["outputs"]["best_pred"] = str(0)
        
    if experiment_type == 1:
        import numpy as np
        ood_pointwise_acc = np.nan
        odd_r2_acc = np.nan
        iid_pointwise_acc = np.nan
        iid_max_r2_acc = np.nan

    elif experiment_type == 2:
        variables = extract_variables_from_infix(str(current["gt"]))
        test_X, test_y = get_robust_random_data(current["gt"], variables)

        ood_pointwise_acc, odd_r2_acc, _ = compute_accuracy_pointwise(test_X,test_y,current["gt"], current["outputs"]["best_pred"], do_sympy_check=do_sympy_check)
        iid_pointwise_acc, iid_max_r2_acc, _ = compute_accuracy_pointwise(current["X"],current["y"],current["gt"], current["outputs"]["best_pred"], do_sympy_check=do_sympy_check)

    cond_satisfactions = []

    if "vanilla" in root_file_name or "standard_nesy" in root_file_name:
        assert current["cond_str"]["condition_str_tokenized"]  == []
        if current["additional_cond_str"] is not None:
            cond_to_match = current["additional_cond_str"]
        else:
            print(f"Warning: additional_cond_str is None for {str(current['gt'])}. Ignoring conditioning")
            cond_to_match = "To be ignored"

        if cond_to_match != "To be ignored":
            # Remove the pointer
            old = cond_to_match["positive_prefix_examples"]
            new = []
            for entry in old:
                for key in entry:
                    if "pointer" in key:
                        break
                else:
                    new.append(entry)
            cond_to_match["positive_prefix_examples"] = new

    else:
        assert current["additional_cond_str"] is None
        cond_to_match = current["cond_str"]

    if experiment_type == 1:
        iterator = tqdm(current["outputs"]['all_raw_preds'])
    else:
        iterator = current["outputs"]['all_raw_preds']

    if "symmetry" in cond_to_match:
        compute_symmetry = True
    else:
        compute_symmetry = False
    for entry in iterator:
        if "illegal" in entry:
            cond_satisfactions.append({"symmetry": 0,"complexity": 0,"positive": 0,"negative": 0,"is_legal": False})
            continue
        if entry == "":
            cond_satisfactions.append({"symmetry": 0,"complexity": 0,"positive": 0,"negative": 0,"is_legal": True})
            continue
        if cond_to_match == "To be ignored":
            import numpy as np
            cond_satisfactions.append({"symmetry": np.nan,"complexity": np.nan,"positive": np.nan,"negative": np.nan,"is_legal": True})
            continue

        properties = compute_properties(entry, compute_symmetry=compute_symmetry,metadata=eq_setting, cfg=cfg)        
        cond_satisfaction = match_properties(properties, cond_to_match)
        cond_satisfaction["is_legal"] = True
        cond_satisfactions.append(cond_satisfaction)

    # Find the index of the best prediction
    if experiment_type == 1:
        best_idx = 0
    else:
        best_raw_pred = current["outputs"]["best_raw_pred"]
        if best_raw_pred is None:
            best_idx = 0
        else:
            best_idx = current["outputs"]['all_raw_preds'].index(best_raw_pred)
        # Swap the best prediction with the first one
        tmp = cond_satisfactions[0]
        cond_satisfactions[0] = cond_satisfactions[best_idx]
        cond_satisfactions[best_idx] = tmp



    if len(cond_satisfactions) == 0:
        import numpy as np 
        cond_satisfactions = [{"symmetry": np.nan,"complexity": np.nan,"positive": np.nan,"negative": np.nan}]

    import pandas as pd
    tmp = pd.DataFrame(cond_satisfactions)
    # Reverse the order of the dataframe
    #tmp = tmp.iloc[::-1]
    symmetry = tmp["symmetry"].values
    complexity = tmp["complexity"].values
    positive = tmp["positive"].values
    negative = tmp["negative"].values
    is_legal = tmp["is_legal"].values


    #cond_satisfaction = accumulate_cond_metrics(cond_satisfactions)

    # Convert default dict to dict
    #cond_satisfaction = {k: v for k, v in cond_satisfaction.items()}

    best_raw_pred = current["outputs"]["best_raw_pred"]
    best_pred = current["outputs"]["best_pred"]
    bfgs_loss = current["outputs"]["best_loss"]


    # if str(current["outputs"]["best_pred"])[:2] == "12":
    #     breakpoint()

    compact_dict = {
        "gt": str(current["gt"]),
        "best_raw_pred": best_raw_pred,
        "best_pred": best_pred,
        "all_preds": current["outputs"]['all_raw_preds'],
        "ood_r2": odd_r2_acc,
        "ood_pointwise_acc": ood_pointwise_acc,
        "iid_r2": iid_max_r2_acc,
        "iid_pointwise_acc": iid_pointwise_acc,
        "bfgs_loss": bfgs_loss,
        "number_of_point": number_of_point,
        "cond_symmetry": symmetry,
        "cond_complexity": complexity,
        "cond_positive": positive,
        "cond_negative": negative,
        "cond_is_legal": is_legal,
        "condition_str_tokenized": str(cond_to_match),
        "original_number_of_point": original_number_of_point,
        "idx": current["idx"],
        "noise": current["noise"],
        "entry": root_file_name,
        "epoch": file.parent.name,
        "folder": file.parent.parent.parent,
        #"is_identical": is_identical,
        "file_name": file.name,
        }
    return compact_dict

import click 

@click.command()
@click.option("--epoch", type=str, default="54")
@click.option("--experiment_type", type=int, default=2)
@click.option("--path_NSRwH", type=str, required=True, default="run/c/2023-07-15/11-39-11")
@click.option("--path_NSR", type=str)
def main(epoch, experiment_type,path_nsrwh, path_nsr):
    path_NSRwH = Path(path_nsrwh)
    weights_path_NSRwH   = path_NSRwH / f"exp_weights/200000000_log_-epoch={epoch}-val_loss=0.00.ckpt"
    cfg_path_NSRwH = omegaconf.OmegaConf.load(path_NSRwH / ".hydra/config.yaml")

    if not path_nsr is None:
        path_NSR = Path(path_nsr)
        weights_path_standard_nesy = f"run/c/2023-01-15/20-07-13/exp_weights/200000000_log_-epoch={epoch}-val_loss=0.00.ckpt"

        
    if experiment_type == 1: # Controllability
        cfg_path_NSRwH.inference.bfgs.activated=False
        cfg_path_NSRwH.inference.beam_size=256
        cfg_path_NSRwH.inference.bfgs.n_restarts=0
        cfg_path_NSRwH.inference.n_jobs = -1
        cfg_path_NSRwH.inference.evaluate = False
        fake_batch = 48
        is_cuda = True
        #noises = [0,0.001,0.0010.1]
        number_of_points_list = [400] #, 200, 100, 50, 25, 10]

    elif experiment_type == 2: # Accuracy
        cfg_path_NSRwH.inference.bfgs.activated=True
        cfg_path_NSRwH.inference.beam_size=5
        cfg_path_NSRwH.inference.bfgs.n_restarts=2
        cfg_path_NSRwH.inference.n_jobs = -1
        fake_batch = -1
        is_cuda = True
        noises = [0.001,0,0.0001,0.01,0.1,1]
        number_of_points_list = [400] #,200,100,50,25,10] #[0.001,0,0.0001,0.01,0.1,1
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
    elif experiment_type == 3: # Matrix 
        cfg_path_NSRwH.inference.bfgs.activated=True
        cfg_path_NSRwH.inference.beam_size=5
        cfg_path_NSRwH.inference.bfgs.n_restarts=2
        cfg_path_NSRwH.inference.n_jobs = -1
        fake_batch = -1
        is_cuda = True
        noises = [0] 
        number_of_points_list = [400] #[0.001,0,0.0001,0.01,0.1,1]
        constants_set = ["100_constant","80_constant","60_constant","40_constant","20_constant","0_constant"]
        positive_set = ["100_positive","80_positive","60_positive","40_positive","20_positive","0_positive"]

        fin_list = []
        for x in constants_set:
            for y in positive_set:
                fin_list.append((x , y))

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    else:
        raise KeyError("experiment_type not implemented")
    

    #folder = path_NSRwH / epoch
    eq_setting = load_metadata_hdf5(Path("configs"))
    

    complete_dicts = []
    compact_dicts = []

    #for folder in path.glob("14"):
        
    # Collect all pickle files from the various folders equal or smaller than the current folder

    #breakpoint()
    nsrwh_folder = (path_NSRwH / Path("res") / epoch)
    if not nsrwh_folder.exists():
        print(f"Warning: {nsrwh_folder} does not exist, provide a valid epoch")
        return None
    
    all_files = list(nsrwh_folder.glob("*.pkl"))
    # Get rid of the _number_number part at the end of the file name
    root_file_names = set(["_".join(x.stem.split("_")[:-2]) for x in all_files])
    
    # Remove seen_eq
    root_file_names = [x for x in root_file_names if not x == "seen"]

    # Add standard_nesy entries to the list
    entries_to_be_changed = [x for x in root_file_names if "vanilla" in x]
    # Replace vanilla with standard_nesy
    entries_to_be_added = [x.replace("vanilla", "standard_nesy") for x in entries_to_be_changed]
    root_file_names = entries_to_be_added + root_file_names

    # Order it such that complexity-aifeymann_processed is the first one
    elem = root_file_names.pop(root_file_names.index("symmetry-only_five_variables_nc"))
    root_file_names = [elem] + root_file_names
    elem = root_file_names.pop(root_file_names.index("vanilla-only_five_variables_nc"))
    root_file_names = [elem] + root_file_names
    
    elem = root_file_names.pop(root_file_names.index("all-aifeymann_processed"))
    root_file_names = [elem] + root_file_names
    elem = root_file_names.pop(root_file_names.index("positive-aifeymann_processed"))
    root_file_names = [elem] + root_file_names
    elem = root_file_names.pop(root_file_names.index("vanilla-aifeymann_processed"))
    root_file_names = [elem] + root_file_names

    #root_file_names = [x for x in root_file_names if ""]
    if experiment_type == 3:
        # Keep only vanilla, full_no_constants and full_constants
        root_file_names = [x for x in root_file_names if "wc" in x and "full_constants" in x]

    for root_file_name in tqdm(root_file_names):
        if "standard_nesy" in root_file_name:
            to_mach = root_file_name.replace("standard_nesy", "vanilla")
        else:
            to_mach = root_file_name
        # Get all files that belong to this root file name
        files = []
        for x in all_files:
            # Check if the prefix of x is the same as root_file_name
            d = len(to_mach)
            if x.stem[:d] == to_mach:
                files.append(x)
        
        # if not "vanilla" in root_file_name:
        #     continue
        if "vanilla" in to_mach:
            additional_data_files = []
            #to_be_added = root_file_name.replace("vanilla", "all")
            additional_data_files = [x.parent / x.name.replace("vanilla", "all") for x in files]

        elif experiment_type == 3:
            files_raw = files 
            files = [file for file in files_raw for _ in fin_list]
            additional_data_files = [x for _ in files_raw for x in fin_list]

        else:
            additional_data_files = []
            #breakpoint()
            #continue

        if "standard_nesy" in root_file_name:
            weights_path = weights_path_standard_nesy 
            cfg_path_NSRwH.architecture.conditioning = False
        else:
            weights_path = weights_path_NSRwH
            cfg_path_NSRwH.architecture.conditioning = "v3"


        if is_cuda:
            fitfunc = return_fitfunc(cfg_path_NSRwH, eq_setting, weights_path, device="cuda")
        else:
            fitfunc = return_fitfunc(cfg_path_NSRwH, eq_setting, weights_path, device="cpu")

        for noise in noises:
            for number_of_point in number_of_points_list:
                for idx_file, file in enumerate(files):
                    # Open file
                    with open(file, "rb") as f:
                        data = pickle.load(f)

                    if additional_data_files != [] and experiment_type != 3: 
                        # Find the file that corresponds to the current file
                        for x in additional_data_files:
                            if x.stem.replace("all", "vanilla") == file.stem:
                                with open(x, "rb") as f:
                                    additional_data = pickle.load(f)
                                    break
                        additional_data_cond_str = additional_data["cond_str"]
                        additional_data_gt = additional_data["gt"]
                    elif experiment_type == 3:
                        assert additional_data_files != []
                        additional = additional_data_files[idx_file]
                    else:
                        additional_data_cond_str = None
                        additional_data_gt = None

                    # if additional == ('100_constant', '100_positive'):
                    #     continue
                    if experiment_type == 3:
                        percent_of_constant = int(additional[0].split("_")[0])
                        percent_of_positive = int(additional[1].split("_")[0])
                        # Recumpute the properties
                        cond_raw = []
                        cond_str = []
                        for idx, gt in enumerate(data["gt"]):
                            curr = sp.sympify(gt)
                            eq_sympy_prefix_constants = Generator.sympy_to_prefix(curr, enable_float=True)
                            cfg.dataset.conditioning.positive.prob_pointers = percent_of_constant / 100
                            cfg.dataset.conditioning.positive.min_percent = percent_of_positive / 100
                            cfg.dataset.conditioning.positive.max_percent = percent_of_positive / 100
                            cfg.dataset.conditioning.positive.prob = 1

                            eq_sympy_prefix_mixed = mix_ptr_constants(eq_sympy_prefix_constants, cfg)
                            
                            positive_prefix_examples, _, cost_to_pointer  = create_positives_and_constants(eq_sympy_prefix_mixed, metadata=eq_setting, cfg=cfg)
                            readable_tokens = []
                            for include in positive_prefix_examples:
                                readable_tokens.extend(["<includes>"] + include + ["</includes>"])
                            
                            cond_str.append(readable_tokens)
                            numerical_tokens = tokenize(readable_tokens, eq_setting.word2id)

                            ordered_cost =[]
                            for key, value in cost_to_pointer.items():
                                ordered_cost.append(float(value))
                            cond_raw.append({"symbolic_conditioning": numerical_tokens, "numerical_conditioning": ordered_cost})

                        data["cond_raw"] = cond_raw
                        data["cond_str"] = cond_str
                        # Pad the elements so they are all the same length
                        max_len = max([len(x["symbolic_conditioning"]) for x in data["cond_raw"]])
                        for idx, cond in enumerate(data["cond_raw"]):
                            cond["symbolic_conditioning"] = cond["symbolic_conditioning"] + [0] * (max_len - len(cond["symbolic_conditioning"]))

                        max_len = max([len(x["numerical_conditioning"]) for x in data["cond_raw"]])
                        for idx, cond in enumerate(data["cond_raw"]):
                            # Pad symbolic conditioning
                            cond["numerical_conditioning"] = cond["numerical_conditioning"] + [0] * (max_len - len(cond["numerical_conditioning"]))

                    current_gt = data["gt"]
                    current_cond_str = data["cond_str"]
                    current_cond_raw = data["cond_raw"]
                    # Convert to tensor each entry of current_cond_raw
                    symbolc_cond = []
                    numerical_cond = []
                    for idx in range(len(current_cond_raw)):
                        s = torch.tensor(current_cond_raw[idx]['symbolic_conditioning'])
                        n = torch.tensor(current_cond_raw[idx]['numerical_conditioning'])
                        symbolc_cond.append(s.unsqueeze(0))
                        numerical_cond.append(n.unsqueeze(0))

                    cond_raw = {"symbolic_conditioning": torch.cat(symbolc_cond,axis=0), "numerical_conditioning": torch.cat(numerical_cond,axis=0)}
                    
                    # Sample a number of points equal to number_of_point
                    original_number_of_point = data["X"].shape[1]
                    if number_of_point < original_number_of_point:
                        pts = torch.randperm(data["X"].shape[1])[:number_of_point]
                    else:
                        
                        pts = torch.arange(original_number_of_point)
                        pts = torch.randperm(len(pts))
                        number_of_point = original_number_of_point
                    X = data["X"][:,pts,:]
                    y = data["y"][:,pts]
                    if noise > 0:
                        import numpy as np
                        for i in range(y.shape[0]):
                            y[i] = y[i] + np.abs(y[i]) * noise * np.random.normal(size=y[i].shape)
                    
                    if is_cuda:
                        X = torch.tensor(X).cuda()
                        y = torch.tensor(y).cuda()
                        cond_input = {"symbolic_conditioning": cond_raw["symbolic_conditioning"].cuda(), "numerical_conditioning": cond_raw["numerical_conditioning"].cuda()}
                    else:
                        X = torch.tensor(X)
                        y = torch.tensor(y)
                        cond_input = {"symbolic_conditioning": cond_raw["symbolic_conditioning"], "numerical_conditioning": cond_raw["numerical_conditioning"]}

                    print(X.shape[1], " points")
                    
                    start = time.time()
                    print("Start fitting")
                    if fake_batch != -1:
                        # Create multiple smaller batches and concatenate them afterward otherwise with big beam size it will crash
                        new_cond_input = {"symbolic_conditioning": [], "numerical_conditioning": []}
                        new_outputss = []
                        iterator = range(0, X.shape[0], fake_batch)
                        for i in tqdm(iterator):
                            new_X = X[i:i+fake_batch]
                            new_y = y[i:i+fake_batch]
                            new_cond_input["symbolic_conditioning"] = cond_input["symbolic_conditioning"][i:i+fake_batch]
                            new_cond_input["numerical_conditioning"] =  cond_input["numerical_conditioning"][i:i+fake_batch]
                            if new_X.shape[0] == 0:
                                break

                            cond_str = current_cond_str[i:i+fake_batch]
                            with torch.no_grad():
                                new_outputs = fitfunc(new_X,new_y,new_cond_input, cond_str, is_batch=True)
                            new_outputss += new_outputs
                    else:
                        with torch.no_grad():
                            new_outputss = fitfunc(X,y,cond_input, current_cond_str, is_batch=True)
                    print("End fitting in ", time.time() - start)
                    
                    if is_cuda:
                        X = X.cpu()
                        y = y.cpu()
                        cond_input = {"symbolic_conditioning": cond_input["symbolic_conditioning"].cpu(), "numerical_conditioning": cond_input["numerical_conditioning"].cpu()}
                    # variables = extract_variables_from_infix(str(current_gt[0]))
                    # from sympy import lambdify
                    # import numpy as np
                    # f = lambdify(variables,str(current_gt[0]), modules=["numpy",{'asin': np.arcsin, "ln": np.log, "Abs": np.abs}])

                    # aaaa = f(*X[0][:,:2].T.half())
                    # #assert len(new_outputss) == len(data["output"].shape[0])
                    entries = []
                    # Prepare the entries
                    cnt = 0
                    print(len(new_outputss))
                    for idx in range(len(new_outputss)):
                        current = {}
                        current["idx"] = idx
                        current["gt"] = current_gt[idx]
                        current["cond_str"] = current_cond_str[idx]
                        current["cond_raw"] = current_cond_raw[idx]
                        current["X"] = X[idx]
                        current["y"] = y[idx]
                        current["outputs"] = new_outputss[idx]
                        current["noise"] = noise
                        
                        if additional_data_gt is not None:
                            try:
                                candidate_idx = [str(x) for x in additional_data_gt].index(str(current_gt[idx]))
                                current["additional_cond_str"] = additional_data_cond_str[candidate_idx]

                            except:
                                candidate_idx = -1
                                print("Not found", current_gt[idx], "in additional data")
                                current["additional_cond_str"] = None
                        else:
                            current["additional_cond_str"] = None
                        
                        entries.append(current)
              
                    print("Computing entries")
                    start = time.time()
                    #
                    res = Parallel(n_jobs=cfg.inference.n_jobs)( 
                                    delayed(return_entries)
                                (
                                    current, root_file_name, file, original_number_of_point=original_number_of_point, eq_setting=eq_setting, cfg=cfg, do_sympy_check=False, experiment_type=experiment_type,
                                ) 
                                for idx, current in tqdm(enumerate(entries)))
                    print("Done in", time.time() - start)
                    res = [x for x in res if x is not None]
                    compact_dicts.extend(res)

                        
                # Same compact_dict 
                import pandas as pd
                df = pd.DataFrame(compact_dicts)
                df.to_pickle(f"results_batched_{folder.stem}_experiment_type_{experiment_type}.pkl")

        
    # Save the complete_dicts
    with open(f"complete_results_batched_{folder.stem}.pkl", "wb") as f:
        pickle.dump(complete_dicts, f)
    breakpoint()

       



if __name__ == "__main__":
    main()