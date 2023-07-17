import math
import random
from itertools import combinations
from pathlib import Path
import json
from typing import List
import warnings

import numpy as np
from numpy import (
    log,
    cosh,
    sinh,
    exp,
    cos,
    tanh,
    sqrt,
    sin,
    tan,
    arctan,
    nan,
    pi,
    e,
    arcsin,
    arccos,
)

import sympy
from sympy.core.rules import Transform
from sympy import sympify, Float, preorder_traversal, lambdify


import torch
from torch.distributions.uniform import Uniform
from torch.utils import data
import pytorch_lightning as pl
import hydra

from ControllableNesymres.utils import load_metadata_hdf5, load_eq
from ControllableNesymres.dclasses import Equation
from ..dataset.data_utils import sample_symbolic_constants, extract_complexity, timeout_return_symmetry, return_symmetry
from ..dataset.generator import Generator, UnknownSymPyOperator
from functools import partial
import copy
import timeout_decorator

### TOKEN WORDS
masking_word = "<mask>"
symmetry_words = [
    "is_not_symmetry",
    "is_symmetry",
    "not_defined"
]

noise_words = [
    "epsilon=0",
    "epsilon=0.001",
    "epsilon=0.01",
    "epsilon=0.1",
    "epsilon=1"
]

special_words = [
    "<includes>",
    "</includes>",
    "<excludes>",
    "</excludes>"
]

float_words = [
    "-",
    "+"
] + ["E"]

pointer_words = [
    "pointer0",
    "pointer1",
    "pointer2",
    "pointer3",
    "pointer4",
    "pointer5",
    "pointer6",
    "pointer7",
    "pointer8",
    "pointer9"
]

# Number words include all numbers from -10.0 to 10.0 in steps of 0.1
number_words = [f"num{str(i/10)}" for i in range(-100, 100, 1)]

MAX_NUM = 65504


def extract_variables_from_prefix(prefix, variables):
    variables = set(variables)
    res = set()
    for x in prefix:
        if x in variables:
            res.add(x)
    return res



def extract_variables_from_infix(equation):
    """ Parse all free variables in equations and return them in
    lexicographic order"""
    expr = sympy.parse_expr(equation)
    variables = expr.free_symbols
    variables = {str(v) for v in variables}
    # # Tighter sanity check: we only accept variables in ascending order
    # # to avoid silent errors with lambdify later.
    # if (variables not in [{'x'}, {'x', 'y'}, {'x', 'y', 'z'}]
    #         and variables not in [{'x1'}, {'x1', 'x2'}, {'x1', 'x2', 'x3'}]):
    #     raise ValueError(f'Unexpected set of variables: {variables}. '
    #                      f'If you want to allow this, make sure that the '
    #                      f'order of variables of the lambdify output will be '
    #                      f'correct.')

    # Make a sorted list out of variables
    # Assumption: the correct order is lexicographic (x, y, z)
    variables = sorted(variables)
    return variables

def replace_ptr_with_costants(raw, ptr_tensor):
    ptr_dict = {pointer_words[idx]:float(x) for idx, x in enumerate(ptr_tensor)}
    # Replace pointers with their values in the raw expression
    for k,v in ptr_dict.items():
        raw = [x if x!=k else v for x in raw]
    if any([x in pointer_words for x in raw]):
        raise ValueError("Pointer not replaced")
    return raw



def description2tokens(description, word2id, cfg):
    """
    Combine the various conditioning together and tokenize them
    """    
    tokens = []

    if "symmetry" in description:
        for sym in description["symmetry"]:
            tokens.append(sym)

    includes = description["positive_prefix_examples"]
    for include in includes:
        tokens.extend(["<includes>"] + include + ["</includes>"])

    #if cfg.dataset.conditioning.include_negative:
    excludes = description["negative_prefix_examples"]
    for exclude in excludes:
        tokens.extend(["<excludes>"] + exclude + ["</excludes>"])

    #if cfg.dataset.conditioning.include_noise:
    if "noise_level" in description:
        tokens.append(description["noise_level"])
    #if cfg.dataset.conditioning.include_complexity:

    if "complexity" in description:
        tokens.append(description["complexity"])
    
    # # Tokenize the sentence
    str_tokens = tokens
    tokens = tokenize(tokens, word2id)
    return tokens, str_tokens

def create_noise_info(info_eq, prob=0.):     
    noise_value = np.random.choice(noise_words)

    info_eq["noise_level_value"] = float(noise_value.split("=")[1])

    is_available = np.random.choice([True, False], p=[prob, 1-prob])
    if is_available:
        info_eq["noise_level"] = noise_value 
    return info_eq

def clean_brackets(positive_example, tmp=[]):
    if type(positive_example) == list or type(positive_example) == tuple:
        for elem in positive_example:
            clean_brackets(elem, tmp)
    else:
        tmp.append(positive_example)

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_positive_or_negative_digit(x):
    tmp = x.isdigit() or (x[0] == "-" and x[1:].isdigit())
    return tmp

def is_token_constant(token):
    return is_float(str(token)) and not is_positive_or_negative_digit(str(token))

def prepare_examples(positive_prefix_examples, negative_prefix_examples, word2id, info=False):        

    # Test whether examples can be tokenized, for avoiding errors later
    tokenized_positive_examples = []
    for prefix in positive_prefix_examples:
        
        try:
            tokenize(prefix, word2id) 
        except:
            if info:
                print("Could not tokenize: {}".format(prefix))
            continue
        tokenized_positive_examples.append(prefix)

    tokenized_negative_examples = []
    for prefix in negative_prefix_examples:
        try:
            tokenize(prefix, word2id)
        except Exception:
            if info:
                print("Could not tokenize: {}".format(prefix))
            continue
        tokenized_negative_examples.append(prefix)
    
    return tokenized_positive_examples,  tokenized_negative_examples


def wrap_complexity(eq_sympy_infix, max_length):
    score = extract_complexity(eq_sympy_infix)
    score = min(max_length, score)
    complexity_word = f"complexity={score}"
    return complexity_word


def create_complexity_info(eq_sympy_infix, prob=0., max_length=25):
    ### warning: the equation might contain placeholders (c) ---> the function is not designed for it
    is_available = np.random.choice([True, False], p=[prob, 1-prob])
    if is_available:
        complexity_word = wrap_complexity(eq_sympy_infix, max_length)
    else:
        complexity_word = None
    return complexity_word


def decompose_equation(prefix_expr, components=[], metadata=None):
    """
    Decompose an equation into its components. Fix behaviour of mul -1. It should be removed
    """
    
    t = prefix_expr[0]
    if t in metadata.operators:
        components.append(t)
        args = []
        l1 = prefix_expr[1:]
        if t in metadata.bin_ops + ["pow"]:
            arity = 2
        elif t in metadata.una_ops + ["abs"]:
            arity = 1
        try:
            for _ in range(arity):  # Arity
                res = decompose_equation(l1,  components=components, metadata=metadata)
                i1, l1 = res
                components.append((t,) + (i1,))
                args.append(i1)
            components.append((t,) + tuple(args))

            return tuple([t] + args), l1
        except:
            print(t)
    elif t in metadata.generator_details.variables:
        components.append(t)
        return t, prefix_expr[1:]
    else: #INT
        val = prefix_expr[0]
        #if not (t.isdigit() or t in {"-1","-2","-3","-4","-5"}):
        components.append(t)
        return str(val), prefix_expr[1:]

def prepare_negative_pool(cfg):
    with open(hydra.utils.to_absolute_path(cfg.path_to_candidate)) as f:
        eq_candidates = json.load(f)

    eq_candidates = [eq.replace(" ", "") for eq in eq_candidates]
    eqs_candidate = sorted(eq_candidates, key=len, reverse=False)
    return eqs_candidate





def return_all_positive_substrees(eq_sympy_prefix,metadata=None, ignore_obvious=False, remove_gt=False):
    components = list() # This contains all the possible candidates
    decompose_equation(eq_sympy_prefix, components=components, metadata=metadata)

    # Remove components
    positive_prefix_examples = set(components)

    # Remove nested levels from the examples
    clean_positive_examples = []
    for positive_example in positive_prefix_examples:
        tmp = []
        clean_brackets(positive_example, tmp=tmp)
        if not tmp in clean_positive_examples:
            clean_positive_examples.append(tmp)
    
    # Order clean_positive_examples by length and alphabetically for being deterministic
    clean_positive_examples = sorted(clean_positive_examples, key=lambda x: (len(x), x))
    if ignore_obvious:
        filtered = []
        for entry in clean_positive_examples:
            if len(entry) == 1:
                x = entry[0]
                if is_positive_or_negative_digit(x):
                    continue
                elif  x in metadata.config["variables"]: 
                    continue
                elif x == "c":
                    continue
                # if is_float(x) and (float(x) == np.inf or float(x) == -np.inf or float(x) == np.nan):
                #     continue
            filtered.append(entry)
        clean_positive_examples = filtered
    
    if remove_gt:
        if [len(x) for x in clean_positive_examples]:
            max_len = max([len(x) for x in clean_positive_examples])
            clean_positive_examples = [x for x in clean_positive_examples if len(x) < max_len]
        else:
            max_len = 0
    return clean_positive_examples

def prepare_pointers(all_positives_examples_mixed):
    pointer_cnt = 0
    cost_to_pointer = {}
    pointer_to_cost = {}
    pointer_examples = []
    all_positive_examples_without_constants = []
    for entry in all_positives_examples_mixed:
        if len(entry) == 1:
            
            candidate_constant = entry[0]
            if  is_token_constant(candidate_constant): #is_float(candidate_constant) and not is_positive_or_negative_digit(candidate_constant): # The token is a constant
                current_pointer = pointer_words[pointer_cnt]
                cost_to_pointer[candidate_constant] = current_pointer
                pointer_to_cost[current_pointer] = candidate_constant
                pointer_cnt += 1
                pointer_examples.append([current_pointer])
            else:
                all_positive_examples_without_constants.append([candidate_constant])
        else:
            all_positive_examples_without_constants.append(entry)

    all_positives_examples_with_ptr = []
    for entry in all_positives_examples_mixed:
        new_entry = []
        for token in entry:
            if token in cost_to_pointer:
                new_entry.append(cost_to_pointer[token])
            else:
                new_entry.append(token)
        all_positives_examples_with_ptr.append(new_entry)

    return all_positives_examples_with_ptr, pointer_examples, pointer_to_cost, pointer_words
    

def create_positives_and_constants(eq_sympy_prefix, metadata, cfg):
    # Convert any constant to str
    eq_sympy_prefix = [str(x) for x in eq_sympy_prefix]
    #start_time = time.time()
    
    all_positives_examples_mixed = return_all_positive_substrees(eq_sympy_prefix, metadata=metadata, ignore_obvious=True, remove_gt=True)
    all_positives_examples_with_ptr, pointer_examples, pointer_to_cost, pointer_words = prepare_pointers(all_positives_examples_mixed)
    len_equation = len(eq_sympy_prefix)


    if len(all_positives_examples_with_ptr) > 0:
        min_percent = cfg.dataset.conditioning.positive.min_percent
        max_percent = cfg.dataset.conditioning.positive.max_percent
        prob = cfg.dataset.conditioning.positive.prob
        if prob > random.random():
            number_of_positive = int(random.uniform(min_percent,max_percent) * len_equation) 
        else:
            number_of_positive = 0

    else:
        number_of_positive = 0
    # Remove from all_positives_examples_with_ptr words that consists only of "pointer" since we are going to add them later
    all_positives_examples_with_ptr = [x for x in all_positives_examples_with_ptr if len(x) > 1 or x[0] not in pointer_words]
    res = return_examples(all_positives_examples_with_ptr,number_of_positive)
    res = pointer_examples + res 
    res = sorted(res, key=lambda x: (len(x), x))
    return res, all_positives_examples_mixed, pointer_to_cost #positive_prefix_examples, all_positives_examples

def should_create_negative(prob):
    return prob > random.random()

def calculate_number_of_negative(min_percent, max_percent, len_equation):
    return int(random.uniform(min_percent, max_percent) * len_equation)

def choose_random_negative_equations(negative_pool, k=4):
    return random.choices(list(negative_pool), k=k)

def create_negatives(eq_sympy_prefix_with_c, negative_pool, all_positives_examples, metadata, cfg):
    min_percent = cfg.dataset.conditioning.negative.min_percent
    max_percent = cfg.dataset.conditioning.negative.max_percent
    prob = cfg.dataset.conditioning.negative.prob
    
    variables = extract_variables_from_prefix(eq_sympy_prefix_with_c, metadata.config["variables"])
    disjoint_variables =  set(metadata.config["variables"]) - variables

    if should_create_negative(prob):
        len_equation = len(eq_sympy_prefix_with_c)
        number_of_negative = calculate_number_of_negative(min_percent, max_percent, len_equation)
    else:
        number_of_negative = 0        

    if number_of_negative>0:
        samples_negative_equations = choose_random_negative_equations(negative_pool, k=cfg.dataset.conditioning.negative.k)
        candidates = []
        for entry in samples_negative_equations:
            _, dummy_consts = sample_symbolic_constants(entry, cfg=None)
        
            eq_string = str((entry.format(**dummy_consts)))
            try:
                eq_sympy_infix_without_constants = sympify_equation(eq_string)
                eq_sympy_infix_without_constants = remove_rationals(eq_sympy_infix_without_constants)
                eq_sympy_infix = replace_constants_with_symbol(eq_sympy_infix_without_constants)
                eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
            except Exception as e:
                continue
                
            all_subtrees = return_all_positive_substrees(eq_sympy_prefix, metadata=metadata, ignore_obvious=True, remove_gt=True)

            # Remove the variables that are not in the equation
            tmp = []
            for cond in all_subtrees:
                if all(x not in disjoint_variables for x in cond):
                    tmp.append(cond)
                    
            tmp = [tuple(x) for x in tmp]
            candidates.extend(tmp)

        raw_candidates_components = set(candidates) - set([tuple(x) for x in all_positives_examples])
        candidate_negative_components = sorted([list(x) for x in raw_candidates_components],  key=lambda x: (len(x), x)) # Sort for being deterministic
        negative_prefix_examples = return_examples(candidate_negative_components, number_of_negative, sampling_type=cfg.dataset.conditioning.negative.sampling_type)
    else:    
        negative_prefix_examples = []
    
    return negative_prefix_examples


def return_examples(available, k, sampling_type=None):
    res = []
    while k > 0:
        available = [x for x in available if k >= len(x)]
        if available == []:
            break

        if sampling_type is None:
            new_candidate = random.sample(available, 1)
        elif sampling_type == "squared":
            weights = [(1/len(x))**2 for x in available] # We want to sample more the shorter ones
            new_candidate = random.choices(available, weights=weights, k=1)
        elif sampling_type == "cubic":
            weights = [(1/len(x))**3 for x in available] # We want to sample more the shorter ones
            new_candidate = random.choices(available, weights=weights, k=1)
        elif sampling_type == "x^4":
            weights = [(1/len(x))**4 for x in available] # We want to sample more the shorter ones
            new_candidate = random.choices(available, weights=weights, k=1)
        else:
            raise ValueError("Sampling type not recognized")
        k = k - len(new_candidate[0])
        available = [x for x in available if not set(x).issubset(set(new_candidate[0])) ]

        curr = []
        
        for x in res:
            if not set(x).issubset(set(new_candidate[0])) :
                curr.append(x)
            else:
                k = k + len(x)
            
            
        res = curr
        res.append(new_candidate[0])

    res = sorted(res, key=lambda x: (len(x), x))

    return res

def return_number_of_examples_negative(cfg):
    if cfg.dataset.conditioning.mode == True:
        min_subtrees, max_subtrees = 0, cfg.dataset.conditioning.max_operators_negative
    else:
        min_subtrees, max_subtrees = 0, 0
    return min_subtrees, max_subtrees


def mix_ptr_constants(eq_sympy_prefix_with_constants, cfg):
    tmp = []
    for entry in eq_sympy_prefix_with_constants:
        prob = cfg.dataset.conditioning.positive.prob_pointers #random.random()
        if type(entry) == sympy.core.numbers.Float: 
            if prob > random.random(): 
                tmp.append(entry)
            else:
                tmp.append("c")
                
        else:
            tmp.append(entry)
    return tmp

   
   

def create_subtrees_info(eq: Equation, negative_pool, metadata=None, cfg=None):
    """
    We need to already tokenize the equation to point out quickly possible issues
    negative_pool: Used for generating the negative examples
    """
    eq_sympy_prefix_mixed = mix_ptr_constants(eq.eq_sympy_prefix_with_constants, cfg)
    positive_prefix_examples, all_positives_examples_mixed, cost_to_pointer  = create_positives_and_constants(eq_sympy_prefix_mixed, metadata, cfg)
    
    # Convert all constants to c for negative examples
    all_positives_examples_without_constants = []
    for cond in all_positives_examples_mixed:
        tmp = []
        for entry in cond:
            if is_token_constant(entry):
                entry = "c"
            tmp.append(entry)
        all_positives_examples_without_constants.append(tmp)
       
    #positive_prefix_examples_with_ptr_and_c = tmp
    assert len(eq.eq_sympy_prefix_with_c) == len(eq.eq_sympy_prefix_with_constants)
    
    negative_prefix_examples = create_negatives(eq.eq_sympy_prefix_with_c, negative_pool, all_positives_examples_without_constants, metadata, cfg)

    positive_prefix_examples, negative_prefix_examples = prepare_examples(positive_prefix_examples, negative_prefix_examples, word2id=metadata.word2id, info=False)

    eq.info_eq["all_positives_examples"] = all_positives_examples_without_constants # Without constants
    eq.info_eq["positive_prefix_examples"] = positive_prefix_examples
    eq.info_eq["negative_prefix_examples"] = negative_prefix_examples
    eq.info_eq["cost_to_pointer"] = cost_to_pointer

    return eq

def preprocess_symmetry(symmetry_raw):
    res = []
    for key in symmetry_raw.keys():
        is_masked = False #np.random.choice([True, False], p=[0*masking_prob/2*0, 1-masking_prob/2])
        if is_masked:
            symmetry_raw[key] = masking_word
        else:
            if symmetry_raw[key] == 0:
                res.append(f"symmetry=NO{key}")
            elif symmetry_raw[key] == 1:
                res.append(f"symmetry={key}")
            elif symmetry_raw[key] == 2:
                pass 
            # elif info_eq["symmetry"][key] == 2:
            #     symmetry_word =  "not_defined"
            else:
                raise KeyError()
    return res


def wrap_symmetry_info( eq_sympy_infix_constants, prob=0.5, variables=None, cached_symmetry=None):
    is_available = np.random.choice([True, False], p=[prob, 1-prob])

    if cached_symmetry is not None:
        symmetry = cached_symmetry 
    else:
        try:
            symmetry = timeout_return_symmetry(str(eq_sympy_infix_constants),variables,n_support=5)
        except timeout_decorator.timeout_decorator.TimeoutError as e:
            print(f"Timeout for {str(eq_sympy_infix_constants)}")
            is_available = False
            symmetry = None
        except Exception as e:
            print(f"Error for {str(eq_sympy_infix_constants)}: Error {e}")
            is_available = False
            symmetry = None
    if symmetry == [] or not is_available:
        return None, symmetry
    else:
        res = preprocess_symmetry(symmetry)
    
    return res, symmetry

def add_conditional_entries_to_word2id(word2id, complexity_words, symmetry_words):
    # Pointer words are the first ones, so that we don't need to increase the softmax size too much
    other_words = pointer_words+ [masking_word] + special_words + symmetry_words + complexity_words + noise_words 
    

    # Add the other tokens to the word2id dict
    for word in other_words:
        if word not in word2id:
            word2id[word] = len(word2id)
    return word2id

def to_scientific_notation(number: float):
    """
    Given a number, returns a string in scientific notation.
    """
    return f"{number:e}"

def replace_constants_with_scientific_notation(eq_sympy_prefix, constants_to_strings):
    """
    Given a sympy prefix, replace the floats with the dictionary of scientific notation strings.
    """
    for key in constants_to_strings:
        # Find the entry in the list that matches key 
        eq_sympy_prefix = [constants_to_strings[key] if type(item) == sympy.core.numbers.Float and float(item) == key else item for item in eq_sympy_prefix]
    
    return eq_sympy_prefix

def create_constants_to_strings(consts):
    """
    Create the mapping between floats and corresponding values
    """
    constants_to_strings = {}          
    for key, value in consts.items():
        if key[:2] == "cm" and value == 1:
            continue
        elif key[:2] == "ca" and value == 0:
            continue
        
        if value > 0:
            sign = "+"
        else:
            sign = "-"
        
        raw_form = to_scientific_notation(value) 
        exponent, significand  = raw_form.split("e")
        exponent = exponent[:4] # Only keep the first 3 digits
        constants_to_strings[value] = (sign, exponent, significand)
    return constants_to_strings

def adapt_conditioning(cfg, name, opt):
    cfg.dataset.conditioning.prob_symmetry = opt["prob_symmetry"]
    cfg.dataset.conditioning.prob_complexity = opt["prob_complexity"]
    cfg.dataset.conditioning.positive.prob = opt["positive.prob"]
    if cfg.dataset.conditioning.positive.prob > 0:
        cfg.dataset.conditioning.positive.min_percent = opt["positive.min_percent"]
        cfg.dataset.conditioning.positive.max_percent = opt["positive.max_percent"]
    cfg.dataset.conditioning.negative.prob = opt["negative.prob"]
    if cfg.dataset.conditioning.negative.prob > 0:
        cfg.dataset.conditioning.negative.min_percent = opt["negative.min_percent"]
        cfg.dataset.conditioning.negative.max_percent = opt["negative.max_percent"]
    if "positive.prob_pointers" in opt:
        cfg.dataset.conditioning.positive.prob_pointers = opt["positive.prob_pointers"]
    cfg.dataset.conditioning.name = name
    cfg.dataset.type_of_sampling_points = "constant"
    return cfg.dataset.conditioning

def return_support_limits(cfg, metadata,support):
    # Sample the eq.support using same method as described in nesymres paper
    if support is not None:
        support_dict = eval(support)
        support_limits = []
        for i in support_dict.keys():
            support_limits.append(Uniform(support_dict[i]["min"], support_dict[i]["max"]))
        return support_limits

    curr_min = cfg.dataset["fun_support"]["min"]
    curr_max = cfg.dataset["fun_support"]["max"]
    min_len = cfg.dataset["fun_support"]["min_len"]
    support_limits = []

    for i in range(len(metadata.config["variables"])): 
        mi = np.random.uniform(curr_min, curr_max-min_len)
        ma = np.random.uniform(mi+min_len, curr_max)
        support_limits.append(Uniform(mi, ma))
    #eq.support = support
    return support_limits

def sample_without_zero(left,right):
    val = random.randint(left, right)
    while val == 0:
        val = random.randint(left, right)
    return val

def has_the_equation_changed(eq,cfg):
    """
    We Use complexity measure to check whether symplification has changed something. This is not perfect, but it is a good approximation.
    """
    before = create_complexity_info(eq.eq_string, prob=1, max_length=cfg.dataset.number_of_complexity_classes)
    after = create_complexity_info(eq.eq_sympy_infix_constants, prob=1, max_length=cfg.dataset.number_of_complexity_classes)
    before = before.split("=")[1]
    after = after.split("=")[1]
    if abs(int(after) - int(before)) > 1:
        return True
    else:
        return False


#timeout(5)
def create_conditioning(eq: Equation,  metadata, word2id, cfg=None,  negative_pool=None):    
    eq = create_subtrees_info(eq, negative_pool=negative_pool, metadata=metadata, cfg=cfg)  

    str_eq_sympy_prefix_with_constants = [str(x) for x in eq.eq_sympy_prefix_with_constants]
    eq.info_eq["ordered_cost"] = []
    
    for key, value in eq.info_eq["cost_to_pointer"].items():
        # Find position in eq_sympy_prefix_constants where value is located
        idx = str_eq_sympy_prefix_with_constants.index(value) # ValueError would be something is wrong in create_subtrees_info
        # Replace in both eq_sympy_prefix and eq_sympy_prefix_constants
        # Make sure eq_sympy_prefix and eq_sympy_prefix_constants are the same but with different values
        assert all([x == y for x, y in zip(eq.eq_sympy_prefix_with_c, str_eq_sympy_prefix_with_constants) if not is_float(y)])

        eq.eq_sympy_prefix_with_c[idx] = key
        eq.eq_sympy_prefix_with_constants[idx] = key
        eq.info_eq["ordered_cost"].append(float(value))


    complexity_word_new = create_complexity_info(eq.eq_sympy_infix_constants, prob=cfg.dataset.conditioning.prob_complexity, max_length=cfg.dataset.number_of_complexity_classes)

    # Symmetry takes a lot of time to compute, so we want to avoid it if possible. We only compute it if the equation has changed
    is_changed = has_the_equation_changed(eq,cfg)
    if is_changed:
        cache = None
    else:    
        cache = eq.info_eq["symmetry"] 

    del eq.info_eq["symmetry"] # Remove the symmetry from the info_eq dict (Coming from the original dataset)

    symmetry, symmetry_raw = wrap_symmetry_info(eq.eq_sympy_infix_constants, prob=cfg.dataset.conditioning.prob_symmetry, variables=metadata.config["variables"], cached_symmetry=cache)
    if symmetry is not None:
        eq.info_eq["symmetry"] = symmetry
    eq.info_eq["symmetry_raw"] = symmetry_raw


    if complexity_word_new is not None:
        eq.info_eq["complexity"] =  complexity_word_new    

    eq.info_eq["condition_tokenized"], eq.info_eq["condition_str_tokenized"]  = description2tokens(eq.info_eq, word2id, cfg=cfg)
    return eq

def word_creator(metadata, cfg):
    xx = []
    rep_combos = np.arange(2,len(metadata.config["variables"]))
    for idx in rep_combos:
        comb = list(combinations(metadata.config["variables"], idx))
        xx.extend(comb)
    
    symmetry_words = [f"{prefix}{x}" for x in xx for prefix in ["symmetry=NO","symmetry="]]
    complexity_words = [f"complexity={str(x)}" for x in range(1,cfg.dataset.number_of_complexity_classes+1)]
    return symmetry_words, complexity_words



def replace_exponents(expr, lower_bound, upper_bound, threshold):
    """
    Replace any exponent larger than the threshold with a random value between the lower and upper bounds.
    """
    for curr in preorder_traversal(expr):
        if isinstance(curr, sympy.Pow):
            if isinstance(curr.args[1], sympy.Float):
                val = sample_without_zero(lower_bound,upper_bound)
                expr= expr.replace(curr.args[1], sympy.Integer(val))
            if isinstance(curr.args[1], sympy.Integer) and abs(curr.args[1]) > threshold:
                val = sample_without_zero(lower_bound,upper_bound)
                expr= expr.replace(curr.args[1], sympy.Integer(val))
    return expr

def replace_small_values(expr, lower_bound, upper_bound, threshold):
    """
    Replace any value smaller than the threshold with a random value between the lower and upper bounds.
    The sign of the value is preserved.
    """
    val = random.uniform(lower_bound, upper_bound)
    expr = expr.xreplace(
        Transform(
            lambda x: sympy.Float(sympy.sign(x) * val) if abs(x) < threshold else x,
            lambda x: isinstance(x, sympy.Float),
        )
    )
    return expr

def replace_large_values(expr, lower_bound, upper_bound, threshold):
    """
    Replace any value bigger than the threshold in absolute value with a random value between the lower and upper bounds.
    The sign of the value is preserved.
    """
    val = random.uniform(lower_bound, upper_bound)
    expr = expr.xreplace(
        Transform(
            lambda x: sympy.Float(sympy.sign(x) * val)if abs(x) > threshold else x,
            lambda x: isinstance(x, sympy.Float),
        )
    )
    return expr


def replace_close_integers(expr, lower_bound, upper_bound, threshold):
    """
    Replace any value close to an integer (within the threshold) with an integer.
    The sign of the value is preserved.
    """
    val = random.uniform(lower_bound, upper_bound)
    expr = expr.xreplace(
        Transform(
            lambda x: sympy.Integer(sympy.sign(x) * val) if isinstance(x, sympy.Float) and abs(x - int(x)) < threshold else x,
            lambda x: isinstance(x, sympy.Float),
        )
    )
    return expr


@timeout_decorator.timeout(5)
def resolve_problematic_constants(expr):
    """
    This function is used to resample constants that have become too small or too big after the previous steps
    """    
    expr = replace_exponents(expr, -4, 4, 5)
    expr = replace_small_values(expr, 0.01, 0.1, 0.001)
    expr = replace_large_values(expr, -5, 5, 15)
    expr = replace_close_integers(expr, -5, 5, 0.005)

    # Redo some steps, because the previous steps might have introduced new problems
    expr = replace_large_values(expr, -5, 5, 15)
    expr = replace_exponents(expr, -4, 4, 5)
    return expr


from torch.distributions.uniform import Uniform

def remove_rationals(expr):
    """
    Remove sympy rationals from the target expression, except 1/2 or -1/2 when they exponent of a power.
    """
    cnt = 0
    for curr in preorder_traversal(expr):
        cnt += 1
        if cnt > 100:
            break
        
        # If the current node is 1/2 or -1/2 and it is the exponent of a power, replace that node with 0.123456789 or -0.123456789 respectively.
        if curr.is_Pow and type(curr.args[1]) in (sympy.core.numbers.Half,sympy.core.numbers.Rational)  and (float(curr.args[1]) == 1/2 or float(curr.args[1]) == -0.5):
            if curr.args[1] == 1/2:
                expr = expr.replace(curr.args[1], float(0.123456789))
            elif curr.args[1] == -0.5:
                expr = expr.replace(curr.args[1], float(-0.123456789))
            
        if isinstance(curr, sympy.core.numbers.Rational) and not isinstance(curr, sympy.Integer):
            number = float(curr)
            expr = expr.subs(curr, sympy.Float(number))
            
    # Replace 1/2 and -1/2 back to sympy rationals
    expr = expr.replace(float(0.123456789), sympy.Rational(1,2))
    expr = expr.replace(float(-0.123456789), sympy.Rational(-1,2))
   
    return expr


def return_costants(eq_sympy_prefix_constants):
    """
    Return the two lists, one with the constants and one with the prefix tree where the constants are replaced with the letter "c"
    """
    eq_sympy_prefix_with_c = []
    res = []
    for x in eq_sympy_prefix_constants:
        if is_token_constant(x):
            eq_sympy_prefix_with_c.append("c")
            res.append(float(x))
        else:
            eq_sympy_prefix_with_c.append(x)
    return res, eq_sympy_prefix_with_c


def sympify_equation(eq_string):
    return sympify(eq_string, evaluate=True)

sympify_equation_timeout = timeout_decorator.timeout(3)(sympify_equation)
remove_rationals_timeout = timeout_decorator.timeout(1)(remove_rationals)

class ControllableNesymresDataset(data.Dataset):
    def __init__(
        self,
        data_path: Path,
        cfg,
        mode: str,
    ):  
        
        metadata = load_metadata_hdf5(data_path)
        self.total_variables = metadata.total_variables
        self.total_coefficients = metadata.total_coefficients

        if mode=="train":
            self.len = cfg.dataset.epoch_len
            self.MAX_ATTEMPTS = 5 # Number of attempts to sample a valid X and Y
        else:
            self.len = metadata.total_number_of_eqs
            self.MAX_ATTEMPTS = 100
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id 
        self.id2word = metadata.id2word
        self.metadata = metadata
        
        # Create the words
        complexity_words, symmetry_words = word_creator(metadata, cfg)

        # Add the other tokens to the word2id dic
        if cfg.dataset.conditioning.mode != False: #"trans"
            self.word2id = add_conditional_entries_to_word2id(self.word2id, complexity_words, symmetry_words)

        self.id2word = {v: k for k, v in self.word2id.items()}
        
        self.data_path = data_path
        self.mode = mode
        
        self.global_dict = {**globals(), **{'asin': np.arcsin, "ln": np.log, "Abs": np.abs, "E": float(sympy.E)}}
        self.cfg = copy.deepcopy(cfg)

        if cfg.dataset.conditioning.mode != False:
            self.negative_pool =  prepare_negative_pool(cfg) 

        if cfg.architecture.predict_constants == False and cfg.dataset.conditioning.positive.prob_pointers > 0:
            raise ValueError("Can't predict constants if networ is not designed for it")
        
    def __getitem__(self, index):
        # Sample a random index in case of training
        if self.mode == "train":
            index = np.random.randint(0, self.metadata.total_number_of_eqs)
       
        # Load the equation from the hdf5
        eq = load_eq(self.data_path, index, self.eqs_per_hdf)
        
        # Sample the constants for the equation
        consts, _ = sample_symbolic_constants(eq, self.cfg.dataset.constants)

        eq_string = eq.expr.format(**consts)

        # Postprocess the equation, by simplifying it and removing constants that are outside the range due to simplification
        try:
            eq_sympy_infix_with_constants = sympify_equation_timeout(eq_string)
            eq_sympy_infix_with_constants = remove_rationals_timeout(eq_sympy_infix_with_constants)
            eq_sympy_infix_with_constants = resolve_problematic_constants(eq_sympy_infix_with_constants)
            eq_sympy_infix_with_constants = sympify_equation_timeout(eq_sympy_infix_with_constants)
            eq_sympy_prefix_with_constants = Generator.sympy_to_prefix(eq_sympy_infix_with_constants, enable_float=True)
        except (OverflowError,TypeError,UnknownSymPyOperator,RecursionError,timeout_decorator.timeout_decorator.TimeoutError)  as e:
            print(f"Equation {eq_string} will be ignored because of:", e)
            return Equation(info_eq=eq.info_eq,code=None,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)

        costants, eq_sympy_prefix_with_c= return_costants(eq_sympy_prefix_with_constants)

        # If any constant is 1e10 or more, something went wrong hence we skip
        if any([np.abs(x) > 1e10 for x in costants]):
            return Equation(info_eq=eq.info_eq,code=None,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)

        eq.eq_string = eq_string
        eq.constants = costants
        eq.eq_sympy_prefix_with_c = eq_sympy_prefix_with_c
        eq.eq_sympy_prefix_with_constants = eq_sympy_prefix_with_constants
        eq.eq_sympy_infix_constants = eq_sympy_infix_with_constants
        # Conditioning section 
        if self.cfg.dataset.conditioning.mode == True:
            try:
                eq = create_conditioning(eq, self.metadata, self.word2id, cfg=self.cfg, negative_pool=self.negative_pool)
            except timeout_decorator.timeout_decorator.TimeoutError as e:
                return Equation(info_eq=eq.info_eq,code=None,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)

            if self.cfg.is_debug:
                to_compare = compute_properties(str(eq.eq_sympy_infix_constants), compute_symmetry=True, metadata=self.metadata, cfg=None)
                
                all_positive_examples = []
                for element in eq.info_eq["all_positives_examples"]:
                    curr = []
                    for token in element:
                        if is_token_constant(token):
                            curr.append("c")
                        else:
                            curr.append(token)
                    all_positive_examples.append(curr)
            
                
                assert sorted(eq.info_eq["all_positives_examples"]) == sorted(all_positive_examples)
                if "complexity" in eq.info_eq:
                    assert eq.info_eq["complexity"] == to_compare["complexity"]
                if "symmetry" in eq.info_eq and eq.info_eq["symmetry"]:
                    assert eq.info_eq["symmetry"] == to_compare["symmetry"]
                       
        try:
            eq.info_eq["target_expr"] = eq_sympy_prefix_with_c
            t = tokenize(eq_sympy_prefix_with_c,self.word2id)
            curr = Equation(info_eq=eq.info_eq, code=None,expr=eq_sympy_prefix_with_c,coeff_dict=consts,variables=eq.variables,support=eq.support, tokenized=t, valid=True)
        except Exception as e:            
            return Equation(info_eq=eq.info_eq, code=None,expr=eq_sympy_prefix_with_c,coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)
        
        variables = list(eq.eq_sympy_infix_constants.free_symbols)
        f = sympy.lambdify(variables, eq.eq_sympy_infix_constants)
        variables_str = [str(x) for x in variables]
        cnt = 0
        while cnt < self.MAX_ATTEMPTS:
            support_limits = return_support_limits(self.cfg, self.metadata, support=eq.support)
            support = sample_support(support_limits, variables_str, self.cfg.dataset.max_number_of_points*5, self.total_variables, self.cfg)
            is_valid, data_points = sample_images(f, support, variables_str, self.cfg)
            if is_valid:
                break
            cnt += 1
            
      
            
        if cnt >= self.MAX_ATTEMPTS:
            return Equation(info_eq=curr.info_eq, code=None,expr=eq_sympy_infix_with_constants,coeff_dict=consts,variables=curr.variables,support=curr.support, valid=False)
        
        else:
            
            # Shuffle the datapoints along the points dimension
            data_points = data_points[:, :, torch.randperm(data_points.shape[2])]

            return Equation(info_eq=curr.info_eq, code=None,expr=eq_sympy_infix_with_constants,coeff_dict=consts,variables=curr.variables,support=curr.support, data_points=data_points, tokenized=t, constants=consts,  valid=True)            
        
    def __len__(self):
        return self.len

def sample_images(lambdify_f, support, variables_str, cfg):
    half_support_used = {}
    for key,value in support.items():
        if key in variables_str:
            half_support_used[key] = value.half()
    # Create a tensor with the support
    support_tensor = []
    for support_row in support.values():
        support_tensor.append(np.array(support_row))

    support_tensor = torch.tensor(np.array(support_tensor)).float()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            aaaa = lambdify_f(**half_support_used)
            if  type(aaaa) == torch.Tensor and aaaa.dtype == torch.float16:
                if torch.bitwise_and(~torch.isnan(aaaa),~(torch.abs(aaaa) >= MAX_NUM)).sum() > cfg.dataset.max_number_of_points:
                    
                    data_points = torch.cat([support_tensor, torch.unsqueeze(aaaa, axis=0)], axis=0).unsqueeze(0)
            
                    # Drop datapoints that have a nan value in the last row (i.e. the y)
                    data_points = data_points[:, :, ~torch.isnan(data_points[0, -1, :])]
                    
                    # Drop datapoints that have a value in the last row (i.e. the y) that is too large
                    data_points = data_points[:, :, torch.abs(data_points[0, -1, :]) < MAX_NUM]
                    
                    data_points = data_points.float()
                    assert data_points.shape[2] > cfg.dataset.max_number_of_points, "Something went wrong in the datapoint generation"
                    return True, data_points
        except NameError as e:
            print(e)
        except RuntimeError as e:
            print(e)

        return False, None


def custom_collate_fn(eqs: List[Equation],  total_variables, total_coefficients, cfg) -> List[torch.tensor]:
    filtered_eqs = [eq for eq in eqs if eq.valid]
    res, tokens, conditioning = evaluate_and_wrap(filtered_eqs, total_variables, total_coefficients, cfg)

    exprs = [x.info_eq['target_expr'] for x in filtered_eqs]
    return res, tokens, [(eq.expr, eq.info_eq) for eq in filtered_eqs], conditioning


def replace_constants_with_symbol(s,symbol="c"):
    sympy_expr = sympify(s)  # self.infix_to_sympy("(" + s + ")")
    sympy_expr = sympy_expr.xreplace(
        Transform(
            lambda x: np.random.uniform(2,3), # We replace with a constant with a dummy value, otherwise sympy might change the expression
            lambda x: isinstance(x, Float),
        )
    )
    return sympy_expr

def match_properties(properties, cond):
    res = {}
    cnt = 0
    if cond is not None and "symmetry" in cond:
        if "symmetry" in properties and len(properties["symmetry"]) > 0:
            for x in cond["symmetry"]:
                if x in properties["symmetry"]:
                    cnt += 1
            acc = cnt / len(properties["symmetry"])
        elif "symmetry" in properties and len(properties["symmetry"]) == 0 and len(cond["symmetry"]) == 0:
            acc = 1
        else:
            acc = 0
        res["symmetry"] = acc
    else:
        res["symmetry"] = np.nan
    # else:
    #     res["symmetry"] = np.nan
    if cond is not None and "complexity" in cond  and "complexity" in properties:
        value_property = int(properties["complexity"].split("complexity=")[1])
        value_condition = int(cond["complexity"].split("complexity=")[1])
        tmp = np.abs(value_property - value_condition)
        res["complexity"] = tmp
    else:
        res["complexity"] = np.nan

    if cond is not None and  cond["positive_prefix_examples"]:
        if "all_positives_examples" in properties:
            cnt = 0
            for x in cond["positive_prefix_examples"]:
                if x in properties["all_positives_examples"]:
                    cnt += 1
            acc = cnt / len(cond["positive_prefix_examples"])
        else:
            acc = 0
        res["positive"] = acc
    
    else:
        res["positive"] = np.nan

    if cond is not None and  cond["negative_prefix_examples"]:
        if "all_positives_examples" in properties:
            cnt = 0
            for x in cond["negative_prefix_examples"]:
                if x in properties["all_positives_examples"]:
                    cnt += 1
            err = 1 - (cnt / len(cond["negative_prefix_examples"]))
        else:
            err = np.nan

        res["negative"] = err #TODO
    else:    
        res["negative"] = np.nan #TODO

    return res

def compute_properties(expr: str, compute_symmetry=False, metadata=None, cfg=None, is_streamlit=False) -> dict:
    """
    Compute all the ground truth properties of the expression
    """
    res = {}

    # Hack replace {constant} with c
    expr = expr.format(constant="c")
    try:
        sympy_expr = sympify(expr, evaluate=True)
    except sympy.SympifyError:
        print("Sympify error with expression: ", expr)
        return res
    

    to_evaluate = expr.format(constant="costant")
    # Replace the constants with random values for the symmetry (Because symmetry requires evaluation)
    possible_values = [0.4, 0.5, 0.8, 1.4, 1.9, 7.1, 9.1,2.3,3.9,4.3,4.8]
    for i, ptr in enumerate(pointer_words):
        to_evaluate = to_evaluate.replace(ptr, str(possible_values[i]), 1)

    appearences = to_evaluate.count("costant")
    possible_values = [1.2, 1.5, 1.8, 3.2, 3.5, 3.8,4.7,5.1,5.3,5.8,6.1]
    for i in range(appearences):
        to_evaluate = to_evaluate.replace("costant", str(possible_values[i]), 1)

    if compute_symmetry:
        try:
            if is_streamlit:
                tmp = return_symmetry(to_evaluate,metadata.config["variables"],n_support=5)
            else:
                tmp = timeout_return_symmetry(to_evaluate,metadata.config["variables"],n_support=5)
            tmp = preprocess_symmetry(tmp)
            res["symmetry"] = tmp
        except Exception as E:
            print("Cannot compute symmetry for expression: ", to_evaluate)
            print("Empty list returned")
    else:
        res["symmetry"] = np.nan
    
    res["complexity"]= wrap_complexity(sympy_expr, max_length=30) #TODO: MAke this pointing #cfg.dataset.number_of_complexity_classes)

    try: 
        pred_sympy_prefix = Generator.sympy_to_prefix(sympy_expr, enable_float=True)
    except Exception as E:
        print("Error when converting sympy to prefix: ", sympy_expr)
        return res

    # Convert anythign that is not a string to a string
    pred_sympy_prefix = [str(x) for x in pred_sympy_prefix]
    clean_positive_examples = return_all_positive_substrees(pred_sympy_prefix, metadata=metadata, ignore_obvious=True, remove_gt=True)
    res["all_positives_examples"] = clean_positive_examples  

    # Negative examples are implicitly defined by the positive ones
    return res
 

def get_robust_random_data(eq, variables, cfg=None):
    #MAX_NUM = 10_000
    MAX_NUM = 65504
    n_attempts_max = 10
    pts = 500
    cnt = 0 

    f = lambdify(variables, eq, modules=["numpy",{'asin': np.arcsin, "ln": np.log, "Abs": np.abs}])
    syms = torch.tensor([])
    aaaas = torch.tensor([])
    while cnt < n_attempts_max:
        if cfg is not None:
            distribution =  torch.distributions.Uniform(cfg.dataset.fun_support.min,cfg.dataset.fun_support.max)
        else:
            distribution =  torch.distributions.Uniform(-25,25) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
        sym = []
        for sy in variables:
            curr = distribution.sample([int(pts*5000)])
            sym.append(curr)
        #try:
        sym = torch.stack(sym)
        # except:
        #     breakpoint()
        input_lambdi = sym
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                aaaa = f(*input_lambdi)
            except NameError as e:
                print(e)
                cnt += 1
                continue
            except RuntimeError as e:
                print(e)
                cnt += 1
                continue
                
            # # Concat with previous good points
            # if syms.shape[0] > 0:
            #     breakpoint()
            #     sym = torch.cat([syms,sym],dim=1)
            #     aaaa = torch.cat([aaaas,aaaa],dim=0)

            if  type(aaaa) == torch.Tensor:
                # Keep good points so far
                syms = sym[:,torch.bitwise_and(~torch.isnan(aaaa),~(torch.abs(aaaa) >= MAX_NUM))]
                aaaas = aaaa[torch.bitwise_and(~torch.isnan(aaaa),~(torch.abs(aaaa) >= MAX_NUM))]
                if len(aaaas) > pts:
                    return syms.T, aaaas
                # else:
                #     # Keep good points so far
                #     syms = sym[:,torch.bitwise_and(~torch.isnan(aaaa),~(torch.abs(aaaa) >= MAX_NUM))]
                #     aaaas = aaaa[torch.bitwise_and(~torch.isnan(aaaa),~(torch.abs(aaaa) >= MAX_NUM))]
                    


            cnt += 1
    raise ValueError("Cannot find good points to sample")

def tokenize(prefix_expr:list, word2id:dict) -> list:
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr:
        try:
            tokenized_expr.append(word2id[i])
        except TypeError:
            pass
    tokenized_expr.append(word2id["F"])
    return tokenized_expr

def de_tokenize(tokenized_expr, id2word:dict):
    prefix_expr = []
    for i in tokenized_expr:
        if "F" == id2word[i]:
            break
        else:
            prefix_expr.append(id2word[i])
    return prefix_expr

def tokens_padding(tokens, ):
    max_len = max([len(y) for y in tokens])
    p_tokens = torch.zeros(len(tokens), max_len)
    for i, y in enumerate(tokens):
        y = torch.tensor(y)
        p_tokens[i, :] = torch.cat([y, torch.zeros(max_len - y.shape[0])])
    return p_tokens

def number_of_support_points(p, type_of_sampling_points):
    if type_of_sampling_points == "constant":
        curr_p = p
    elif type_of_sampling_points == "logarithm":
        curr_p = int(10 ** Uniform(1, math.log10(p)).sample())
    elif type_of_sampling_points == "uniform":
        curr_p = int(Uniform(1, p).sample())
    else:
        raise NameError
    return curr_p

def sample_support(support_limits, variables, curr_p, total_variables, cfg):
    #sym = []
    sym_dict = {}
    if not support_limits:
        distribution =  torch.distributions.Uniform(cfg.fun_support.min,cfg.fun_support.max) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
        
        for sy in total_variables:
            if sy in variables:
                curr = distribution.sample([int(curr_p)])
            else:
                curr = torch.zeros(int(curr_p))
            sym_dict[sy] = curr
            #sym.append(curr)
    else:
        #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
        
        for idx, sy in enumerate(total_variables):
            if sy in variables:
                # try:
                distribution = support_limits[idx]
                curr = distribution.sample([int(curr_p)])
            else:
                curr = torch.zeros(int(curr_p))
            sym_dict[sy] = curr
            #sym.append(curr)

    return sym_dict #torch.stack(sym)

def sample_constants(eq, curr_p, total_coefficients):
    consts = []
    #eq_c = set(eq.coeff_dict.values())
    for c in total_coefficients:
        if c[:2] == "cm":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.ones([int(curr_p)])
        elif c[:2] == "ca":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.zeros([int(curr_p)])
        consts.append(curr)
    return torch.stack(consts)

def evaluate_and_wrap(eqs: List[Equation], total_variables, total_coefficients, cfg):
    vals = []
    cond0 = []
    tokens_eqs = [eq.tokenized for eq in eqs]
    if cfg.dataset.conditioning.mode != False:
        symbolic_conditioning = [eq.info_eq["condition_tokenized"] for eq in eqs]
        costants_eqs = [eq.info_eq["ordered_cost"] for eq in eqs]


    tokens_eqs = tokens_padding(tokens_eqs)
    curr_p = number_of_support_points(cfg.dataset.max_number_of_points, cfg.dataset.type_of_sampling_points)
    vals = [x.data_points[:,:,:curr_p] for x in eqs]

    num_tensors = torch.cat(vals, axis=0)

    if cfg.dataset.conditioning.mode != False:
        symbolic_conditioning = tokens_padding(symbolic_conditioning)
        costants_eqs = tokens_padding(costants_eqs)
    else:
        symbolic_conditioning = []
        costants_eqs = []
        
    return num_tensors, tokens_eqs, {"symbolic_conditioning": symbolic_conditioning, "numerical_conditioning": costants_eqs}


CONFIG_PLACEHOLDER = { "prob_symmetry": 0, "prob_complexity": 0, "positive.prob": 0,  "negative.prob":0, "positive.prob_pointers": 0}
dataloader_configs = {}
val_dataloader_keys = {"vanilla","complexity","symmetry","noise","constants","positive","negative","constants_and_positive","all","full_constants","full_no_constants"}
for key in val_dataloader_keys:
    if key == "vanilla":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
    elif key == "complexity":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["prob_complexity"] = 1
    elif key == "symmetry":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["prob_symmetry"] = 1
    elif key == "constants": # Only single ptr 
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["positive.prob"] = 0 # But it will extract constants
        dataloader_configs[key]["positive.min_percent"] = 0.5
        dataloader_configs[key]["positive.max_percent"] = 0.5
        dataloader_configs[key]["positive.prob_pointers"] = 0.8
    elif key == "positive":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["positive.prob"] = 1
        dataloader_configs[key]["positive.min_percent"] = 0.5
        dataloader_configs[key]["positive.max_percent"] = 0.5
    elif key == "negative":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["negative.prob"] = 1
        dataloader_configs[key]["negative.min_percent"] = 0.5
        dataloader_configs[key]["negative.max_percent"] = 0.5
    elif key == "constants_and_positive":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["positive.prob"] = 1
        dataloader_configs[key]["positive.min_percent"] = 0.5
        dataloader_configs[key]["positive.max_percent"] = 0.5
        dataloader_configs[key]["positive.prob_pointers"] = 0.3
    elif key == "all":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["prob_complexity"] = 1
        dataloader_configs[key]["prob_symmetry"] = 1
        dataloader_configs[key]["positive.prob"] = 1
        dataloader_configs[key]["positive.min_percent"] = 0.5
        dataloader_configs[key]["positive.max_percent"] = 0.5
        dataloader_configs[key]["negative.prob"] = 1
        dataloader_configs[key]["negative.min_percent"] = 0.5
        dataloader_configs[key]["negative.max_percent"] = 0.5
        dataloader_configs[key]["positive.prob_pointers"] = 0.3
    elif key == "full_constants":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["prob_complexity"] = 1
        dataloader_configs[key]["prob_symmetry"] = 1
        dataloader_configs[key]["positive.prob"] = 1
        dataloader_configs[key]["positive.min_percent"] = 1
        dataloader_configs[key]["positive.max_percent"] = 1
        dataloader_configs[key]["negative.prob"] = 1
        dataloader_configs[key]["negative.min_percent"] = 1
        dataloader_configs[key]["negative.max_percent"] = 1
        dataloader_configs[key]["positive.prob_pointers"] = 1
    elif key == "full_no_constants":
        dataloader_configs[key] = CONFIG_PLACEHOLDER.copy()
        dataloader_configs[key]["prob_complexity"] = 1
        dataloader_configs[key]["prob_symmetry"] = 1
        dataloader_configs[key]["positive.prob"] = 1
        dataloader_configs[key]["positive.min_percent"] = 1
        dataloader_configs[key]["positive.max_percent"] = 1
        dataloader_configs[key]["negative.prob"] = 1
        dataloader_configs[key]["negative.min_percent"] = 1
        dataloader_configs[key]["negative.max_percent"] = 1
        dataloader_configs[key]["positive.prob_pointers"] = 0
    




class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_train_path,
        data_benchmark_path,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.data_train_path = data_train_path
        self.data_benchmark_path = data_benchmark_path
        self.training_dataset = ControllableNesymresDataset(
                    self.data_train_path,
                    self.cfg.copy(),
                    mode="train"
                )
        

    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)    

    def setup(self, stage=None):
        """called one ecah GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:
            if self.data_train_path:    
                self.validation_datasets = []
                candidates = sorted(list(self.data_benchmark_path.glob("*")))
                for path in candidates:
                    if path.is_dir():
                        for name, opt in dataloader_configs.items():
                            if not "wc" in path.name and "constant" in name:
                                continue
                                
                            # Prepare the config for the dataset
                            curr_cfg = self.cfg.copy()
                            curr_cfg.dataset.conditioning = adapt_conditioning(curr_cfg, name, opt)

                            if self.data_benchmark_path:
                                self.validation_dataset = ControllableNesymresDataset(
                                    path,
                                    curr_cfg,
                                    mode="val",
                                )
                            

                            self.validation_datasets.append([self.validation_dataset, curr_cfg, [path.name, name ]])
    def train_dataloader(self):
        """returns training dataloader"""
        print("Test")
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.batch_size//self.cfg.gpu,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(custom_collate_fn, total_variables=self.training_dataset.total_variables, total_coefficients=self.training_dataset.total_coefficients, cfg= self.cfg),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            #worker_init_fn=self.worker_init_fn
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        mapper = {}
        validation_dataloader = []
        
        for idx, curr in enumerate(self.validation_datasets):
            dataset, curr_cfg, dataloader_info = curr
            validloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                collate_fn=partial(custom_collate_fn,total_variables=self.training_dataset.total_variables, total_coefficients=self.training_dataset.total_coefficients, cfg= curr_cfg),
                num_workers=min(0,self.cfg.num_of_workers),
                pin_memory=True,
                drop_last=False,
                worker_init_fn=self.worker_init_fn
            )
            validation_dataloader.append(validloader)
            opt, name = dataloader_info 

            mapper[idx] =  f"{name}-{opt}"
            validation_dataloader

        self.mapper = mapper
        return validation_dataloader

    def test_dataloader(self):
        """returns validation dataloader"""
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(custom_collate_fn,cfg=self.cfg.dataset_test),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=False
        )

        return testloader
