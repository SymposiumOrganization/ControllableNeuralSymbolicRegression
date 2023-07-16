from pmlb import fetch_data
from pathlib import Path
import omegaconf
from ControllableNesymres.utils import load_metadata_hdf5
import click
from ControllableNesymres.utils import return_fitfunc
import torch
from ControllableNesymres.architectures import utils
import sympy as sp
import pandas as pd
import numpy as np
from ControllableNesymres.architectures.utils import get_variables, evaluate_func,get_pointwise_acc
import time 
from sklearn.model_selection import train_test_split
from ControllableNesymres.dataset.generator import Generator
import pickle 
from ControllableNesymres.architectures.data import tokenize, create_positives_and_constants, is_float, compute_properties, is_float, is_positive_or_negative_digit
import random


def read_pmlb_file(
    filename: str, label: str = "target", nrows: int = 9999, sep: str = None):
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    input_data = pd.read_csv(
        filename, sep=sep, compression=compression, nrows=nrows, engine="python"
    )
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]
    return X, y, feature_names

@click.command()
@click.option("--epoch", type=str, default="149")
@click.option("--experiment", type=str, default="complexity_iteration")
@click.option("--path", type=str, required=True)
def main(epoch,experiment,path):
    path = Path(path)
    root_path = path.parent
    cfg =  omegaconf.OmegaConf.load(Path(root_path / ".hydra/config.yaml"))

    cfg.inference.bfgs.activated=True
    
    cfg.inference.bfgs.n_restarts=10
    cfg.inference.n_jobs = -1
    cfg.inference.evaluate = True
    cfg.inference.rejection_sampling = False
    cfg.inference.rejection_sampling_n_samples = 0
    cfg.inference.target_metric = "r2_val" 
    sample_temperature = None
    
    is_cuda = False
    if is_cuda:
        device = "cuda"
    else:
        device = "cpu"

    if experiment == "positive_exp":
        # Load some equations from the dataset
        seen_eqs = set()
        folder = Path("run/c")
        for file in folder.glob("**/seen_eqs_*"):
            with open(file, "rb") as f:
                data = pickle.load(f)
            seen_eqs.update(data)
            break
        eq_setting = load_metadata_hdf5("")

        cfg.dataset.conditioning.positive.prob_pointers = 0
        cfg.dataset.conditioning.positive.min_percent = 50 / 100
        cfg.dataset.conditioning.positive.max_percent = 50 / 100
        cfg.dataset.conditioning.positive.prob = 1
        # CUDA VISIBLE DICE 2
        import os 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        priors = set()
        # Check if a file name "priors.pkl" exists
        file_exist = False
        if Path("assets/priors_new.pkl").is_file():
            with open("assets/priors_new.pkl", "rb") as f:
                priors = pickle.load(f)
            file_exist = True
        
        if not file_exist:
            
            while True:
                prefix = random.choice(list(seen_eqs))
                if "x_3" in prefix or "x_4" in prefix or "x_5" in prefix:
                    continue

                
                #curr = sp.sympify(gt)
                #eq_sympy_prefix_constants = Generator.sympy_to_prefix(curr, enable_float=True)
                positive_prefix_examples, _, cost_to_pointer  = create_positives_and_constants(prefix, metadata=eq_setting, cfg=cfg)
                
                
                # Remove priors with "digits" such as 2, 3, 4
                to_keep = []
                for entry in list(positive_prefix_examples):
                    if "3" in entry or "4" in entry or "5" in entry or "-3" in entry or "-4" in entry or "-5" in entry:
                        continue
                    if "pointer0" in entry or "pointer1" in entry or "pointer2" in entry:
                        continue
                    to_keep.append(entry)
                positive_prefix_examples_tup = [tuple(x) for x in to_keep]
                priors.update(positive_prefix_examples_tup)
                print(len(priors))
                if len(priors) > 1000:
                    break
            with open("assets/priors_new.pkl", "wb") as f:
                pickle.dump(priors, f)
        priors = list(priors)
        

    elif experiment == "complexity_exp":
        total_beam_size = 250
        smaller_beam_size = 10
        complexities = [x for x in range(1,25)]
        iterations = [[]] + complexities
        batch_size = len(complexities)
        #assert len(iterations)*smaller_beam_size == total_beam_size
        total_beam_size = 1

    elif experiment == "gomea":
        gomea_csv = pd.read_csv("assets/gomea_resultsv2.csv")

    elif experiment == "last_hope":
        gomea_csv = pd.read_csv("assets/gomea_resultsv2.csv")
        iterations = [x for x in range(5)]
        # Sample up to 15 
        max_number_of_positive = int(15) 
    else:
        raise NotImplementedError


   
    folder = path / epoch
    eq_setting = load_metadata_hdf5(cfg.train_path )
    weights_path = f"run/c/2023-01-15/20-06-17/exp_weights/200000000_log_-epoch={folder.stem}-val_loss=0.00.ckpt"
    cfg.architecture.conditioning = True

    eqs = ['712_chscase_geyser1', '687_sleuth_ex1605', '601_fri_c1_250_5', '596_fri_c2_250_5', 
             '529_pollen', '1096_FacultySalaries', '628_fri_c3_1000_5', '624_fri_c0_100_5', '656_fri_c1_100_5', 
               '1030_ERA', '613_fri_c3_250_5', '690_visualizing_galaxy', '631_fri_c1_500_5', 
            '597_fri_c2_500_5', '612_fri_c1_1000_5', '678_visualizing_environmental', 'banana',
             '649_fri_c0_500_5', '594_fri_c2_100_5', '609_fri_c0_1000_5', '617_fri_c3_500_5', 
             '599_fri_c2_1000_5', '579_fri_c0_250_5', '192_vineyard', '611_fri_c3_100_5', '228_elusage',
            'strogatz_bacres2', 'strogatz_barmag1', 'strogatz_barmag2', 'strogatz_glider1', 'strogatz_glider2', 'strogatz_lv1', 
            'strogatz_lv2', 'strogatz_predprey1', 'strogatz_predprey2', 'strogatz_shearflow1', 'strogatz_shearflow2', 'strogatz_vdp1', 'strogatz_vdp2']

    
    if is_cuda:
        fitfunc = return_fitfunc(cfg, eq_setting, weights_path, device="cuda")
    else:
        fitfunc = return_fitfunc(cfg, eq_setting, weights_path, device="cpu")
    
    
    res = []
    tensor_batch = []
    read_sentences = []
    for idx_num, seed in enumerate([1,2,3,4,5,6,7,43923,933414,43123,984321,11111,910321,15795,34343,12341,11284,11964]): #, 11964, 11964, 15795, 15795]):
        for idx_eq, eq in enumerate(eqs):
        

            if experiment == "positive_exp":
                # Sample 100 - len(first_round) elements from the second round
                random.shuffle(priors)
                current_conditioning = priors[:1]
                iterations = current_conditioning #["None"]  + [[]] + current_conditioning
                batch_size = 1 #499 #len(current_conditioning)
                total_beam_size = 10
                smaller_beam_size = 10
                #assert len(iterations[1:])*smaller_beam_size <= total_beam_size
            best_scores = {}
            if not "strogatz" in eq:
                continue

            print("iterating over eq ", eq)
            try:  
                X_raw, y_raw = fetch_data(eq, return_X_y=True)
            except:
                X_raw, y_raw = read_pmlb_file(f"pmlb/datasets/{eq}.tsv.gz" ,nrows=1000)
            
            print("X max", np.max(X_raw))
            print("X min", np.min(X_raw))
            dim_eq = X_raw.shape[1]
            if experiment == "last_hope":
                # Get the correct equation 
                
                gomea_eq = gomea_csv[gomea_csv["dataset"] == eq]
                # Get eq with the same seed
                gomea_eq_same_seed = gomea_eq[gomea_eq["random_state"] == seed]
                eq_found = gomea_eq_same_seed["symbolic_model"].values[0]
                # Replace pi with 3.141592653589793
                eq_found = eq_found.replace("Pi","3.141592653589793")
                
                # Remove any spurious p in the equation
                eq_found = eq_found.replace("p/","/")
                # Replace plog with log
                eq_found = eq_found.replace("plog","log")
                # Replace x0 with x_0 and x1 with x_1, etc
                for i in range(5):
                    eq_found = eq_found.replace(f"x{i}",f"x_{i+1}")

                sympy_expr = sp.sympify(eq_found, evaluate=True)

                properties = compute_properties(eq_found, compute_symmetry=False,metadata=eq_setting, cfg=cfg)   
                
                # Convert all constants to c for negative examples
                all_positives_examples_without_constants = []
                for cond in properties["all_positives_examples"]:
                    tmp = []
                    for entry in cond:
                        if is_float(entry) and not is_positive_or_negative_digit(entry):
                            entry = "c"
                        tmp.append(entry)
                    all_positives_examples_without_constants.append(tmp)
                
                # Remove duplicates entries
                all_positives_examples_without_constants_with_c = [list(x) for x in set(tuple(x) for x in all_positives_examples_without_constants)]

                # Remove c from the list of positives
                new_list = []
                for entry in all_positives_examples_without_constants_with_c:
                    if len(entry) == 1 and entry[0] == "c":
                        continue
                    new_list.append(entry)
                all_positives_examples_without_constants = new_list
               
                
            # Keep some points for testing
            train_idxs, test_idxs = train_test_split(np.arange(len(X_raw)), train_size=0.75, test_size=0.25, random_state=seed)
            X_test_ = X_raw[test_idxs]
            y_test_ = y_raw[test_idxs]    
            X_train = X_raw[train_idxs]
            y_train = y_raw[train_idxs]

            # Select 40% of the points for validation
            train_idxs, val_idxs = train_test_split(np.arange(len(X_train)), train_size=0.60, test_size=0.40, random_state=seed+1)
            X_val_ = X_train[val_idxs]
            y_val_ = y_train[val_idxs]
            X_train = X_train[train_idxs]
            y_train = y_train[train_idxs]

            if X_train.shape[0] > 400:
                # Randomly sample 1000 points
                idxs = np.random.choice(np.arange(X_train.shape[0]), size=400, replace=False)
                X_train = X_train[idxs]
                y_train = y_train[idxs]


            for sentence in iterations:
                X = torch.tensor(X_train).unsqueeze(0)
                y = torch.tensor(y_train).unsqueeze(0)

                X_test = torch.tensor(X_test_).unsqueeze(0)
                y_test = torch.tensor(y_test_).unsqueeze(0)

                X_val = torch.tensor(X_val_).unsqueeze(0)
                y_val = torch.tensor(y_val_).unsqueeze(0)
                

                columns_to_add = 5 - X.shape[2] 
                if columns_to_add > 0:
                    columns_to_add = torch.zeros(X.shape[0],X.shape[1],columns_to_add)
                    columns_to_add_2 = torch.zeros(X_test.shape[0],X_test.shape[1],columns_to_add.shape[2])
                    X = torch.cat([X,columns_to_add],dim=2)
                    X_test = torch.cat([X_test,columns_to_add_2],dim=2)
                
                
                if experiment in ["positive_exp","last_hope"]:
                    read_sentence = []
                    tokenized_sentence = []
                    if sentence != "None" and sentence != []:
                        read_sentence.extend(["<includes>"] + list(sentence) + ["</includes>"])
                        tokenized_sentence = tokenize(read_sentence, eq_setting.word2id)
                        #except:
                        #continue
                    else:
                        tokenized_sentence = [1,2]

                    
                elif experiment == "complexity_exp":
                    if sentence != [] and sentence != "None":
                        assert sentence in range(0,30)
                        token = eq_setting.word2id[f"complexity={sentence}"]
                        tokenized_sentence.append(token)
                        read_sentence = [f"complexity={sentence}"]
                    
                if sentence == "None":
                    cfg.inference.beam_size=total_beam_size
                    weights_path = "run/c/2023-01-15/20-07-13/exp_weights/200000000_log_-epoch=149-val_loss=0.00.ckpt" 
                    cfg.architecture.conditioning = False
                    fitfunc = return_fitfunc(cfg, eq_setting, weights_path, device=device)
                else:
                    weights_path = "run/c/2023-01-15/20-06-17/exp_weights/200000000_log_-epoch=149-val_loss=0.00.ckpt" 
                    cfg.architecture.conditioning = True
                    if cfg.inference.beam_size != smaller_beam_size:
                        cfg.inference.beam_size=smaller_beam_size
                        fitfunc = return_fitfunc(cfg, eq_setting, weights_path, device=device)
                cond_input = {"symbolic_conditioning": torch.tensor([tokenized_sentence]), "numerical_conditioning": torch.tensor([[]])}
                    
                if is_cuda:
                    X = X.cuda()
                    y = y.cuda()
                    cond_input["symbolic_conditioning"] = cond_input["symbolic_conditioning"].cuda()
                    cond_input["numerical_conditioning"] = cond_input["numerical_conditioning"].cuda()
                
                if sentence != "None":
                    read_sentences.append(read_sentence)
                    tensor_batch.append(cond_input)
                    if len(tensor_batch) < batch_size:
                        continue
                    else:
                        # Padd the tensor batch to the same length
                        max_len = max([x["symbolic_conditioning"].shape[1] for x in tensor_batch])
                        new_symbolic_cond = []
                        new_numerical_cond = []
                        for i in range(len(tensor_batch)):
                            current_symbolic_cond = tensor_batch[i]["symbolic_conditioning"]
                            to_add = torch.zeros(1,max_len-current_symbolic_cond.shape[1]).long()
                            if is_cuda:
                                to_add = to_add.cuda()
                            current_symbolic_cond = torch.cat([current_symbolic_cond,to_add],dim=1)
                            entry = {"symbolic_conditioning": current_symbolic_cond, "numerical_conditioning": torch.tensor([[]])}
                            if is_cuda:
                                entry["symbolic_conditioning"] = entry["symbolic_conditioning"].cuda()
                                entry["numerical_conditioning"] = entry["numerical_conditioning"].cuda()
                            new_symbolic_cond.append(entry["symbolic_conditioning"])
                            new_numerical_cond.append(entry["numerical_conditioning"])

                        
                        new_symbolic_cond = torch.cat(new_symbolic_cond,dim=0)
                        new_numerical_cond = torch.cat(new_numerical_cond,dim=0)
                        new_cond_input = {"symbolic_conditioning": new_symbolic_cond, "numerical_conditioning": new_numerical_cond}
                        X = X.repeat(len(tensor_batch),1,1)
                        y = y.repeat(len(tensor_batch),1)
                        X_val = X_val.repeat(len(tensor_batch),1,1)
                        y_val = y_val.repeat(len(tensor_batch),1)
                        cond_input = new_cond_input
                        
                        
                else:
                    read_sentences = [["None"]]

                        
                        

                start_time = time.time()
                with torch.no_grad():
                    new_outputss = fitfunc(X,y,cond_input, val_X=X_val, val_y=y_val, sample_temperature=sample_temperature, is_batch=True)
                end_time = time.time()

                all_none_dict = { "r2_test": np.nan, "r2_train": np.nan,"r2_val":np.nan, "err_val": np.nan, "err_train": np.nan, "err_val": np.nan,   "pointwise": np.nan, "prediction": np.nan, "sorted_idx": np.nan}
                time_taken = end_time - start_time
                for idx, output in enumerate(new_outputss):
                    prediction = output["best_pred"]
                    
                    #prediction = "1.920855+0.309135*((((cos(x1)+(x0-5.247000))+sin((x0+x1)))-(((x0*x0)+cos(x2))-((x1+x4)+(x3+x3)))))"
                    # Sympify the prediction
                    read_sentence = read_sentences[idx]
                    if read_sentence == []:
                        read_sentence = "None_Cond"
                    if prediction == None or str(prediction) == "nan":
                        if read_sentence == []:
                            read_sentence = "None_Cond"
                        else:
                            read_sentence = "_".join(read_sentence)
                        curr  = {"read_sentence": read_sentence,  "seed":seed, "eq_name":eq,"idx_eq": idx_eq, "dim_eq":dim_eq, "idx": idx, "time": time_taken}
                        curr = {**curr, **all_none_dict}
                        res.append(curr)
                        #df = pd.DataFrame(res)
                        #df.to_csv(f"result_blackbox.csv")
                        continue

                    try:
                        prediction = sp.sympify(prediction)
                        variables = get_variables(str(prediction))
                    except:
                        read_sentence = "_".join(read_sentence)
                        curr  = {"read_sentence": read_sentence, "seed":seed, "eq_name":eq,"idx_eq": idx_eq, "dim_eq":dim_eq, "idx": idx, "time": time_taken}
                        curr = {**curr, **all_none_dict}
                        res.append(curr)
                        #df = pd.DataFrame(res)
                        #df.to_csv(f"result_blackbox.csv")
                        continue
                    # Lambify the prediction
                    #breakpoint()
                    
                    
                    #variables = ['x0', 'x1', 'x2', 'x3', 'x4']
                    curr_X_val = X_val[0:1].squeeze().numpy()
                    curr_X_test = X_test[0:1].squeeze().numpy()
                    curr_X = X[0:1].squeeze().cpu().numpy()
                    try:
                        curr_v_pred_val = evaluate_func(str(prediction),variables, curr_X_val[:,:len(variables)]).numpy()
                        curr_y_pred_test = evaluate_func(str(prediction),variables, curr_X_test[:,:len(variables)]).numpy()
                        curr_y_pred = evaluate_func(str(prediction),variables, curr_X[:,:len(variables)]).numpy()
                    except NameError:
                        read_sentence = "_".join(read_sentence)
                        curr  = {"read_sentence": read_sentence, "seed":seed, "eq_name":eq,"idx_eq": idx_eq, "dim_eq":dim_eq, "idx": idx, "time": time_taken}
                        curr = {**curr, **all_none_dict}
                        res.append(curr)
                        df = pd.DataFrame(res)
                        df.to_csv(f"result_blackbox.csv")
                        continue
                    # variables = get_variables(str(gt))
                    # variables = np.sort([str() list(prediction.free_symbols))
                    # f = sp.lambdify(variables, prediction)
                    curr_y_val = y_val[0:1].squeeze().numpy()
                    curr_y_test = y_test[0:1].squeeze().numpy()
                    curr_y = y[0:1].squeeze().cpu().numpy()
                    err_test = np.mean((curr_y_test - curr_y_pred_test)**2)
                    err_val = np.mean((curr_y_val - curr_v_pred_val)**2)
                    err_train = np.mean((curr_y - curr_y_pred)**2)
                    r2_val = utils.stable_r2_score(curr_y_val, curr_v_pred_val)
                    r2_test = utils.stable_r2_score(curr_y_test, curr_y_pred_test)
                    r2_train = utils.stable_r2_score(curr_y, curr_y_pred)
                
                    pointwise_acc = get_pointwise_acc(curr_y_test, curr_y_pred_test,rtol=0.05,atol=0.001)
                    # print(f"R2: {r2_acc}")
                    # print(f"Pointwise: {pointwise_acc}")
                    if read_sentence == []:
                        read_sentence = "None_Cond"
                    elif read_sentence == "None":
                        read_sentence = "None"
                    else:
                        read_sentence = "_".join(read_sentence)

                    curr  = {"read_sentence": read_sentence, "seed":seed, "eq_name":eq, "idx_eq": idx_eq, "dim_eq":dim_eq, "idx": idx, 
                                "r2_test": r2_test, "r2_train": r2_train,"r2_val":r2_val, 
                                "err_val": err_test, "err_train": err_train, "err_val": err_val, 
                                "pointwise": pointwise_acc, "prediction": prediction, 
                                "sorted_idx": output["sorted_idx"], "all_losses_val": output["all_losses"], 
                                "time": time_taken}
                    res.append(curr)
                start = time.time()
                df = pd.DataFrame(res)
                df.to_pickle(f"result_only_small_black_box.pkl")
                end = time.time()
                print(f"Time taken to save: {end-start}")
                read_sentences = []
                tensor_batch = []
        
        
    

if __name__ == "__main__":
    main()