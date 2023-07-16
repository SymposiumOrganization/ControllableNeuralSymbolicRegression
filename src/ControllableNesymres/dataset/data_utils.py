from http.client import TEMPORARY_REDIRECT
from typing import Tuple
import torch
from torch._C import Value
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import numpy as np
import random
import warnings
import inspect
from torch.distributions.uniform import Uniform
import math
import types
from numpy import log, cosh, sinh, exp, cos, tanh, sqrt, sin, tan, arctan, nan, pi, e, arcsin, arccos
from sympy import sympify,lambdify, Symbol,simplify, preorder_traversal
from sympy import Float
from ..dclasses import Equation
#from .utils import load_dataset
import numpy as np
import pandas as pd
import sympy
import timeout_decorator 
import sympy
import itertools
import warnings


def create_uniform_support(sampling_distribution, n_variables, p):
    sym = {}
    for idx in range(n_variables):
        sym[idx] = sampling_distribution.sample([int(p)])
    support = torch.stack([x for x in sym.values()])
    return support


def group_symbolically_indetical_eqs(data,indexes_dict,disjoint_sets):
    for i, val in enumerate(data.eqs):
        if not val.expr in indexes_dict:
            indexes_dict[val.expr].append(i)
            disjoint_sets[i].append(i)
        else:
            first_key = indexes_dict[val.expr][0]
            disjoint_sets[first_key].append(i)
    return indexes_dict, disjoint_sets


def dataset_loader(train_dataset, test_dataset, batch_size=1024, valid_size=0.20):
    num_train = len(train_dataset)
    num_test_h = len(test_dataset)
    indices = list(range(num_train))
    test_idx_h = list(range(num_test_h))
    np.random.shuffle(test_idx_h)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )
    test_loader_h = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, valid_loader, test_loader_h, valid_idx, train_idx


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def initialize_weights(m):
    """Used for the transformer"""
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate_fun(args):
    """Single input algorithm as this function is used in multiprocessing"""
    fun ,support = args
    if type(fun)==list and not len(fun):
        return []

    global_dict = {**globals(), **{'asin': np.arcsin, "ln": np.log, "Abs": np.abs, "E": float(sympy.E)}}
    f = types.FunctionType(fun, globals=global_dict, name='f')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            evaled = f(*support)
            if type(evaled) == torch.Tensor and evaled.dtype == torch.float32:
                return evaled.numpy().astype('float16')
            else:
                return []
    except NameError as e:
        print(e)
        return []
    except RuntimeError as e:
        print(e)
        return []

def return_dict_metadata_dummy_constant(metadata):
    dict = {key:0 for key in metadata.total_coefficients}
    for key in dict.keys():
        if key[:2] == "cm":
            dict[key] = 1
        elif key[:2] == "ca":
            dict[key] = 0
        else:
            raise KeyError
    return dict
    

def sp_expr_to_skeleton(expr):
    constants = {}
    for i, f in enumerate(expr.atoms(sympy.Float)):
        r = sympy.Symbol(f"C{i}", real=True)
        expr = expr.subs(f, r)
        constants[f"C{i}"] = f
    return expr, constants


def return_dummy_consts(eq_infix):
    # Count the number cm_ and ca_ constants in the equation
    number_of_cm = eq_infix.count("cm_")
    number_of_ca = eq_infix.count("ca_")

    multiplicative_candidates = [f"cm_{x}" for x in range(0, number_of_cm)]
    additive_candidates = [f"ca_{x}" for x in range(0, number_of_ca)]

    dummy_consts = {}
    
    for c in multiplicative_candidates:
        if c in eq_infix:
            dummy_consts[c] = 1

    for c in additive_candidates:
        if c in eq_infix:
            dummy_consts[c] = 0


    return dummy_consts



def sample_symbolic_constants(eq, cfg=None) -> Tuple:
    """Given an equation, returns randomly sampled constants and dummy constants.
    Args:
        eq : An equation object. Can be of type Equation or type str.
        cfg (Optional): Configuration object used for specifying the number and range of constants to sample.
            If None, the sampled constants will be the same as the dummy constants.

    Returns:
        Tuple: A tuple containing two dictionaries.
            - consts: A dictionary of constants sampled from the equation with their corresponding values.
            - dummy_consts: A dictionary of dummy constants from the equation. Dummy constants are 1 if they are multiplicative and 0 if they are additive.
    """
    if type(eq) == str:
        dummy_consts = return_dummy_consts(eq)
        consts = dummy_consts.copy()
    elif  type(eq).__name__ == Equation.__name__:
        dummy_consts = {const: int(const.startswith("cm")) for const in eq.coeff_dict}
        consts = dummy_consts.copy()
    else:
        raise TypeError(f"eq must be of type Equation or str, got {type(eq)}")

    if cfg is not None and cfg.enabled and type(eq) == Equation:
        num_constants = min(len(eq.coeff_dict),cfg.num_constants)
        candidates = list(range(0,num_constants))
        if candidates:
            used_consts = random.choices(candidates, weights=[1/(i+1) for i in candidates], k=1)[0]
        else:
            used_consts = 0
        
        symbols_used = random.sample(list(eq.coeff_dict.keys()), used_consts)
        for symbol in symbols_used:
            if symbol.startswith("ca"):
                consts[symbol] = round(float(Uniform(cfg.additive.min, cfg.additive.max).sample()),3)
            elif symbol.startswith("cm"):
                # Sample from log-uniform distribution the multiplicative constants
                min_log = math.log(cfg.multiplicative.min)
                max_log = math.log(cfg.multiplicative.max)
                sampled = float(Uniform(min_log, max_log).sample())
                consts[symbol] = round(math.exp(sampled),3)
            else:
                raise KeyError
    elif cfg is not None and cfg.enabled and type(eq) == str:
        raise NotImplementedError
    return consts, dummy_consts


def all_combinations(any_list):
    return itertools.chain.from_iterable(
        itertools.combinations(any_list, i + 1)
        for i in range(len(any_list)))




#@timeout_decorator.timeout(5) 
def return_symmetry(eq,symbols,n_support=5):
    if 'Abs' in eq:
        eq = eq.replace('Abs','sin')
    combos = list(all_combinations(np.arange(len(symbols))))
    combos_ = list(all_combinations([str(s) for s in symbols]))
    combos = combos[len(symbols):-1]
    combos_ = combos_[len(symbols):-1]
    results = {str(i):2 for i in combos_}
    grads = {s: sympy.diff(eq,s) for s in symbols}
    grad_squared = {s: grads[s]**2 for s in symbols}
    for iii, combo in enumerate(combos):
        if not False in [i in eq for i in combos_[iii]]:
            symm = 0
            symb = [symbols[i] for i in combo]
            grad = [grads[x] for x in symb]
            denominator = sympy.sqrt(sum([grad_squared[x] for x in symb]))
            #norm_grad = [(grad[i]/denominator) for i in range(len(grad))]
            comple = list(set(symbols) - set(symb))
            comple_comb = list(all_combinations(np.arange(len(comple))))[-1:]

            for cc in comple_comb:
                symbs = [comple[i] for i in cc]
                eval = np.ones((n_support,len(symb)))/100
                eval = np.array([eval[:,i]*(2.5*i+1) for i in range(len(symb))]).T
                eval = np.concatenate([np.random.rand(n_support,len(symbs)),eval], axis=1)
                n = lambdify(symbs+symb, denominator,modules=["sympy", {'sqrt': sympy.sqrt, 'Abs': sympy.Abs}])
                norm_values = np.array([n(*i) for i in eval])
                for ii in range(len(grad)):
                    f = lambdify(symbs+symb, grad[ii],modules=["sympy", {'sqrt': sympy.sqrt, 'Abs': sympy.Abs}])
                    values = np.array([f(*i) for i in eval])
                    to_check = np.abs(np.diff(np.abs(values/norm_values)))
                    if np.any(to_check)<1e-15:
                        symm=1
                        break
                if symm == 1:
                    break
            results[str(combos_[iii])]=symm
        else:
            continue
    return results

timeout_return_symmetry = timeout_decorator.timeout(5)(return_symmetry)

def count_nodes(expr):
    node_count = 0
    for node in preorder_traversal(expr):
        node_count += 1
    return node_count

def extract_complexity(expr):
    #expr = simplify(expr)
    eq = str(expr)
    if 'pi' in str(eq):
        eq = eq.replace('pi','3.14')
    if 'E' in str(eq):
        eq = eq.replace('E',str(np.exp(1)))
    if 'I' in str(eq):
        eq = eq.replace('I',str(1))
    try:
        score=count_nodes(sympify(eq))
    except Exception as e:
        return "unknown"
    return score


