from dataclasses import dataclass
from types import CodeType
from typing import List, Tuple
from torch.distributions import Uniform, Normal, Distribution
from dataclass_dict_convert import dataclass_dict_convert
import torch
import sympy

@dataclass
class Equation:
    info_eq: dict
    code: CodeType
    expr: str 
    coeff_dict: dict
    variables: list #FIXME
    support: tuple = None
    tokenized: list = None
    valid: bool = True
    number_of_points: int = None
    data_points: torch.Tensor = None
    constants: dict = None
    eq_string: str = None # Raw string of the equation before any symplification
    eq_sympy_prefix_with_c: list = None
    eq_sympy_prefix_with_constants: list = None # Same as eq_sympy_prefix but with the actual constant instead of c
    eq_sympy_infix_constants: sympy.Expr = None # Sympy expression with c


@dataclass
class GeneratorDetails:
    max_len: int
    operators: str
    max_ops: int
    #int_base: int
    #precision: int
    rewrite_functions: str
    variables: list
    eos_index: int
    pad_index: int

@dataclass
class DatasetDetails:
    #eqs: List[Equation]
    config: dict
    total_coefficients: list
    total_variables: list
    word2id: dict
    id2word: dict
    una_ops: list
    bin_ops: list
    operators: list
    rewrite_functions: list 
    total_number_of_eqs: int
    eqs_per_hdf: int
    generator_details: GeneratorDetails
    unique_index: set = None
    


@dataclass
class BFGSParams:
    activated: bool = True
    n_restarts: bool = 10
    add_coefficients_if_not_existing: bool = False
    normalization_o: bool = False
    idx_remove: bool = True
    normalization_type: str = ["MSE","NMSE"][0]
    stop_time: int = 14

@dataclass
class FitParams:
    word2id: dict
    id2word: dict
    total_coefficients: list
    total_variables: list
    rewrite_functions: list
    una_ops: list = None
    bin_ops: list = None
    bfgs: BFGSParams = BFGSParams()
    beam_size: int = 2
    n_jobs: int = 1
    evaluate: bool = True
    rejection_sampling: bool = False
    rejection_sampling_n_samples: int = 5
    metadata: dict = None
    target_metric: str = "mse"
