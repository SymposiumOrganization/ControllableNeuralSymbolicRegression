import numpy as np
from scipy.optimize import minimize
import torch
import sympy as sp
import time
import sympy
import re
from ..architectures.data import extract_variables_from_infix
import warnings
import timeout_decorator
import threading

class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    
    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value

def return_c0(s, loss, x0, cfg):
    fun_timed = TimedFun(fun=sp.lambdify(s,loss, modules=["numpy", {'asin': np.arcsin, "ln": np.log, "Abs": np.abs}]), stop_after=cfg.bfgs.stop_time)
    if len(x0):
        minimize(fun_timed.fun,x0, method='BFGS')   #check consts interval and if they are int
        return fun_timed
    else:
        return []
    
return_c0_timeout = timeout_decorator.timeout(15)(return_c0)





def bfgs(candidate, X, y,  cfg=None):

    #Check where dimensions not use, and replace them with 1 to avoid numerical issues with BFGS (i.e. absent variables placed in the denominator)
    y = y.squeeze()
    X = X.clone().half()
    
    #bool_dim = (X==0).all(axis=1).squeeze()

    vars_list = extract_variables_from_infix(candidate)
    # Remove "constant" from vars_list
    vars_list = [x for x in vars_list if x != "constant"]

    if len(vars_list) == 0:
        print("Candidate expression is a constant: ", candidate)
        print("Adding artifically a variable x_0")
        vars_list = ["x_0"]

    indeces = [int(x[2:])-1 for x in vars_list]
    
    X = X[:,:,indeces] #= 1 
    
    c = 0 
    expr = candidate
    for i in range(expr.count("constant")):
        expr = expr.replace("constant", f"c{i}",1)
    # if len(vars_list)==2:
    #     breakpoint()
    
    symbols = {i: sp.Symbol(f'c{i}') for i in range(candidate.count("constant"))}   
    #if cfg.bfgs.activated:
    if len(symbols) > 0:

        if cfg.bfgs.idx_remove:
            bool_con = (X<200).all(axis=2).squeeze() 
            X = X[:,bool_con,:]
            y = y[bool_con]
            # idx_leave = np.where((np.abs(input_batch[:,3].numpy()))<200)[0]
            # xx = xx[:,idx_leave]
            # input_batch = input_batch[idx_leave,:]


        # max_y = np.max(np.abs(torch.abs(y).cpu().numpy()))
        # print('checking input values range...')
        # if max_y > 300:
        #     print('Attention, input values are very large. Optimization may fail due to numerical issues')

        diffs = []
        for i in range(X.shape[1]):
            curr_expr = expr
            for index, vars in zip(indeces, vars_list):
                curr_expr = sp.sympify(curr_expr).subs(vars,X[:,i,index]) 
            diff = curr_expr - y[i]
            diffs.append(diff)


        loss = 0
        cnt = 0
        # print(expr)
        # if len(vars_list)==2:
        #     breakpoint()
        if cfg.bfgs.normalization_type == "NMSE": # and (mean != 0):
            mean_y = np.mean(y.numpy())
            if abs(mean_y) < 1e-06:
                print("Normalizing by a small value")
            loss = (np.mean(np.square(diffs)))/mean_y  ###3 check
        elif cfg.bfgs.normalization_type == "MSE": 
            loss = (np.mean(np.square(diffs)))
        
        # elif cfg.bfgs.normalization_type is None:
        #     loss = sum(diffs)
        else:
            raise KeyError
        
        # If the loss contains nan, inf or complex values return nan 
        if sp.I in sp.sympify(loss).atoms():
            return np.nan, np.nan, np.nan, np.nan

        # Lists where all restarted will be appended
        F_loss = []
        consts_ = []
        funcs = []
        
        pseudo_symbols = [value for key, value in symbols.items()]
        lambified_loss = sp.lambdify(pseudo_symbols, loss)

        # Sample a random tensor of 10000 elements between -10 and 10 of size equal to the number of constants
        constant_tensor = torch.rand(100000, len(symbols)).half() * (2000) - 1000
        start = time.time()
        try:
            value = lambified_loss(*constant_tensor.T)
        except RuntimeError:
            print("Error in evaluating the loss function")
            return np.nan, np.nan, np.nan, np.nan
        print("Time to evaluate the loss function: ", time.time() - start)
        number_of_attempts = cfg.bfgs.n_restarts
        # Return the index of the k smallest elements in value
        try:
            sorted_t, idx_t = value.sort()
        except:
            print("Error in sorting")
            return np.nan, np.nan, np.nan, np.nan
        candidates = constant_tensor[idx_t[:number_of_attempts]]
        for i in range(cfg.bfgs.n_restarts):
            # Compute number of coefficients
            x0 = candidates[i]


            s = list(symbols.values())

           
            #bfgs optimization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    if threading.current_thread() is threading.main_thread():
                        fun_timed = return_c0_timeout(s, loss, x0, cfg)
                    else:
                        fun_timed = return_c0(s, loss, x0, cfg)
                    consts_.append(fun_timed.x)
                except timeout_decorator.timeout_decorator.TimeoutError:
                    consts_.append([])
                    continue

            final = expr
            #final = final.replace('ln','log') #### new luca ##TODO: Similar to other problems, Check if it needs a fix. Fixed Should be ok commented
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(len(s)):
                    final = sp.sympify(final).replace(s[i],fun_timed.x[i])
            if cfg.bfgs.normalization_o:
                funcs.append(max_y*final)
            else:
                funcs.append(final)
            
            #values = {x:X[:,:,idx].cpu().half() for idx, x in enumerate(cfg.total_variables)} #CHECK ME
            values = {}
            for idx, vars in zip(indeces, vars_list):
                values[vars] = X[:,:,idx].cpu().half()

            # Disable warnings for the following line
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    y_found = sp.lambdify(vars_list, final, modules=["numpy",{'asin': np.arcsin, "ln": np.log, "Abs": np.abs, "E": float(sympy.E)}])(**values)
                except RuntimeError:
                    print("Runtime error with expr {}".format(final))
                    y_found = torch.zeros(y.shape)

            final_loss = np.mean(np.square(y_found-y.cpu()).numpy())
            
            F_loss.append(final_loss)

            # if len(vars_list)==2:
            #     breakpoint()
        try:
            k_best = np.nanargmin(F_loss)
        except ValueError:
            print("All-Nan slice encountered")
            k_best = 0
        return funcs[k_best], consts_[k_best], F_loss[k_best], expr
    else: # No constants in the expression
        raise ValueError("This should not happen")
     

