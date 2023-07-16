import numpy as np
from collections import defaultdict
import copy
import warnings
import torch
from sklearn.metrics import r2_score
import copy 
from sklearn.exceptions import UndefinedMetricWarning
from collections import defaultdict
from sympy import sympify, simplify
from timeout_decorator import timeout
import timeout_decorator 
from ..dclasses import Equation
#from .utils import load_dataset
import numpy as np
from sympy import sympify
import pandas as pd
import sympy
import timeout_decorator 
import itertools
import numpy as np
import sympy
from sympy import lambdify, sympify, parse_expr
import torch

from ..dataset.data_utils import timeout_return_symmetry

@timeout(1)
def quick_check(gt,candidate):
    res = simplify(sympify(gt) - sympify(candidate))
    if res == 0:
        return True
    else:
        return False

def compute_accuracy_pointwise(X, y, gt, candidate, do_sympy_check=False):
    # X_dim = self.cfg.architecture.dim_input - 1
    # points_for_eval = torch.rand(size=(20, X_dim))*10
    variables = get_variables(str(gt))
    #pointwise_accs, r2_accs, exprs  = [], [], []


    
    # for idx, expr in enumerate(candidates[:until]):
    try:
        y_pred = evaluate_func(candidate,get_variables(candidate), X[:,:len(variables)])
    except Exception as e:
        print("Exception {}".format(e))
        print("Issue evaluating {}. GT: {}. Should not matter".format(candidate, str(gt)))
        max_pointwise_acc = np.nan
        max_r2_accs = np.nan 
        idx = np.nan 
        expr = ""
        return max_pointwise_acc, max_r2_accs, expr
    
    pointwise_acc = get_pointwise_acc(y, y_pred,rtol=0.05,atol=0.001)

    if do_sympy_check:
        try:
            res = quick_check(gt,candidate)
        except timeout_decorator.timeout_decorator.TimeoutError as e:
            res = False
    else:
        res = False
    if res == True and pointwise_acc < 0.999:
        breakpoint()
    
    r2_acc = stable_r2_score(y, y_pred)
    return pointwise_acc, r2_acc, candidate


def accumulate_r2_metrics(score_values):
    results_scores = np.array(score_values)
    results_scores = np.clip(results_scores, 0, 1)
    scores = {}
    scores[f"r2-mean"] = np.nanmean(results_scores)
    scores[f"r2-median"] = np.nanmedian(
        results_scores
    )
    scores[f"r2>0.5"] = np.mean(
        results_scores > 0.9
    )
    scores[f"r2>0.9"] = np.mean(
        results_scores > 0.9
    )
    scores[f"r2>0.99"] = np.mean(
        results_scores > 0.99
    )
    scores[f"r2>0.999"] = np.mean(
        results_scores > 0.999
    )
    scores[f"r2>0.9999"] = np.mean(
        results_scores > 0.9999
    )

    return scores

def accumulate_cond_metrics(cond_satisfactions):
    res = defaultdict(list)
    for entry in cond_satisfactions:
        for k, v in entry.items():
            res[k].append(v)
    for k, v in res.items():
        res[k] = np.nanmean(v)

    return res

def stable_r2_score(y, y_tilde):
    y_true = copy.deepcopy(y)
    y_pred = copy.deepcopy(y_tilde)
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"Got y {y_true.shape} and {y_pred.shape}"
    try:
        is_okey = (np.isnan(y_true) != np.isnan(y_pred)).any()
    except TypeError:
        return -np.inf
    if (np.isnan(y_true) != np.isnan(y_pred)).any():
        return -np.inf
    elif (np.isinf(y_true) != np.isinf(y_pred)).any():
        return -np.inf
    elif (np.isinf(y_true) & (np.sign(y_true) != np.sign(y_pred))).any():
        return -np.inf
    to_remove = np.isnan(y_true) | np.isinf(y_true)
    y_true_finite = y_true[~to_remove]
    y_pred_finite = y_pred[~to_remove]
    if y_true_finite.shape[0] == 0 and y_pred_finite.shape[0] == 0:
        return 1.0
    if (y_true_finite == y_pred_finite).all():
        return 1.0
    elif y_true_finite.shape[0] > 0 and y_pred_finite.shape[0] > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return r2_score(y_true_finite, y_pred_finite, force_finite=True)
            except RuntimeWarning:
                return -np.inf
            except ValueError:
                return -np.inf
            except UndefinedMetricWarning:
                return -np.inf
    else:
        return -np.inf

def float2bit(f, device=None, num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.float32):
        ## SIGN BIT
        s = (torch.sign(f+0.001)*-1 + 1)*0.5 #Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        ## EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2**(num_e_bits-1)-1)
        e_decimal = e_scientific + (2**(num_e_bits-1)-1)
        e = integer2bit(e_decimal, device, num_bits=num_e_bits)
        ## MANTISSA
        f2 = f1/2**(e_scientific)
        m2 = remainder2bit(f2 % 1, device, num_bits=bias)
        fin_m = m2[:,:,:,:num_m_bits] #[:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

def remainder2bit(remainder, device=None, num_bits=127):
    dtype = remainder.type()
    exponent_bits = torch.arange(num_bits, device = device).type(dtype)
    exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
    out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
    return torch.floor(2 * out)

def integer2bit(integer,device=None, num_bits=8):
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1, device = device).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2



def get_pointwise_acc(y_true, y_pred, rtol, atol):
    # r = roughly_equal(y_true, y_pred, rel_threshold)
    try:
        r = np.isclose(y_true, y_pred, rtol=rtol, atol=atol, equal_nan=True)
    except TypeError:
        r = np.array([0,0]) # If there is an error, we return 0
    
    return r.mean()

def all_combinations(any_list):
    return itertools.chain.from_iterable(
        itertools.combinations(any_list, i + 1)
        for i in range(len(any_list)))



def get_data(eq: Equation,number_of_points,  mode):
    """
    iid_ood_mode: if set to "iid", sample uniformly from the support as given
                  by supp; if set to "ood", sample from a larger support

    """
    sym = []
    vars_list = []
    for i, var in enumerate(eq.variables):
        # U means random uniform sampling.
        # Currently this is the only mode we support.
        # Decide what to do about the linspace mode.
        # assert 'U' in eq.support[var], (f'Support modes {eq.support[var].keys()} not '
        #                             f'implemented! Decide what to do about '
        #                             f'those.')
        l, h = eq.support[var]["min"], eq.support[var]["max"]
        if mode == 'iid':
            x = np.random.uniform(l, h,number_of_points)
        elif mode == 'ood':
            support_length = h - l
            assert support_length > 0
            x = np.random.uniform(l-support_length, h+support_length,
                                    number_of_points)
        else:
            raise ValueError(f'Invalid iid_ood_mode: {mode}')
        sym.append(x)
        vars_list.append(vars_list)

    X = np.column_stack(sym)
    assert X.ndim == 2
    assert X.shape[0] == number_of_points
    var = return_order_variables(eq.variables)
    y = evaluate_func(eq.expr, var, X)
    #y = lambdify(var,eq.expr)(*X.T)[:,None]
    #y = evaluate_func(gt_equation, vars_list, X)
    return X, y




def get_robust_data(eq: Equation,mode): # Maybe we could use this for complexity
    n_attempts_max = 100
    X, y = get_data(eq,  eq.number_of_points, mode)
    for _ in range(n_attempts_max):
        to_replace = np.isnan(y).squeeze() | np.iscomplex(y).squeeze()
        if not to_replace.any():
            break

        n_to_replace = to_replace.sum()
        X[to_replace,:], y[to_replace] = get_data(eq,n_to_replace,mode)
    if to_replace.any():
        #get_data(eq,  eq.number_of_points, mode)
        raise ValueError('Could not sample valid points for equation '
                         f'{eq.expr} supp={eq.support}')
        
        
    return X, y




    
def load_equation(benchmark_path, equation_idx):
    df = load_data(benchmark_path)
    benchmark_row = df.loc[equation_idx]
    gt_equation = benchmark_row['eq']
    supp = eval(benchmark_row['support'])
    variables = set(supp.keys())
    ie = timeout_return_symmetry(gt_equation,['x_1','x_2','x_3','x_4','x_5'])  ##### to change
    eq = Equation(info_eq = ie,  
                    code=None, 
                    expr=gt_equation, 
                    coeff_dict= None, 
                    variables=variables, 
                    support=supp, 
                    valid = True,
                    number_of_points= benchmark_row['num_points'] )
    return eq

def load_data(benchmark_name):
    df = pd.read_csv(benchmark_name)
    if not all(x in df.columns for x in ["eq","support","num_points"]):
        raise ValueError("dataframe not compliant with the format. Ensure that it has eq, support and num_points as column name")
    df = df[["eq","support","num_points"]]
    return df    


def get_variables(equation):
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

def return_order_variables(var:set):
    return sorted(list(var), key= lambda x: int(x[2:]))



def evaluate_func(func_str, vars_list, X):
    assert X.ndim == 2
    assert len(set(vars_list)) == len(vars_list), 'Duplicates in vars_list!'

    order_list = vars_list
    indeces = [int(x[2:])-1 for x in order_list]

    if not order_list:
        # Empty order list. Constant function predicted
        expr = str(parse_expr(func_str))
        f = lambdify([], expr, modules=["numpy",{'asin': np.arcsin, "ln": np.log, "Abs": np.abs}])
        try:
            res = f() * np.ones(X.shape[0])
        except Exception as e:
            res = np.ones(X.shape[0])
        return res

    # Pad X with zero-columns, allowing for variables to appear in the equation
    # that are not in the ground-truth equation
    X_padded = np.zeros((X.shape[0], len(vars_list)))

    
    X_padded[:, :X.shape[1]] = X[:,:X_padded.shape[1]]
    # Subselect columns of X that corrspond to provided variables
    try:
        X_subsel = X_padded[:, indeces]
    except:
        print('ERROR with variables')
        X_subsel = X_padded[:, -X_padded.shape[1]:]

    # The positional arguments of the resulting function will correspond to
    # the order of variables in "vars_list"
    
    # replace ln with log
    #func_str = func_str.replace('ln','log')

    expr = str(parse_expr(func_str))
    f = lambdify(vars_list, expr,  modules=["numpy",{'asin': np.arcsin, "ln": np.log, "Abs": np.abs}])

    res = f(*torch.tensor(X_subsel).T)
    return res 



def evaluate_func_(func_str, vars_list, X):
    order_list = vars_list
    indeces = [int(x[2:])-1 for x in order_list]

    if not order_list:
        # Empty order list. Constant function predicted
        f = lambdify([], func_str)
        return f() * np.ones(X.shape[0])

    # Pad X with zero-columns, allowing for variables to appear in the equation
    # that are not in the ground-truth equation
    X_padded = np.zeros((X.shape[0], len(vars_list)))

    
    X_padded[:, :X.shape[1]] = X[:,:X_padded.shape[1]]
    # Subselect columns of X that corrspond to provided variables
    try:
        X_subsel = X_padded[:, indeces]
    except:
        print('ERROR')
        X_subsel = X_padded[:, -X_padded.shape[1]:]

    # The positional arguments of the resulting function will correspond to
    # the order of variables in "vars_list"
    f = lambdify(vars_list, func_str)
    return f(*X_subsel.T)