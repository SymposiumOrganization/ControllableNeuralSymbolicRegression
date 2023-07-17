import streamlit as st
from ControllableNesymres.utils import return_fitfunc
import omegaconf
from pathlib import Path
from ControllableNesymres.utils import load_metadata_hdf5, retrofit_word2id
import numpy as np
import pandas as pd
from ControllableNesymres.architectures.data import compute_properties, create_negatives,\
                                                    prepare_negative_pool, sympify_equation,\
                                                    return_costants, description2tokens, prepare_pointers,\
                                                    tokenize, is_token_constant, get_robust_random_data, return_support_limits,sample_support,sample_images
import base64
import streamlit as st
from ControllableNesymres.dataset.generator import Generator
import sympy
import torch
import random

### Streamlit utitlity functions
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: white;
        ">
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="800" 
                    height="425" 
                    type="application/pdf" 
                    style="background-color: white;"
            ></iframe>
        </div>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


def update_multiselect_style():
    st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb="tag"] {
                height: fit-content;
            }
            .stMultiSelect [data-baseweb="tag"] span[title] {
                white-space: normal; max-width: 100%; overflow-wrap: anywhere;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_selectbox_style():
    st.markdown(
        """
        <style>
            .stSelectbox [data-baseweb="select"] div[aria-selected="true"] {
                white-space: normal; overflow-wrap: anywhere;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
### End of Streamlit utility functions




def main():
    # Set wide layout
    st.set_page_config(layout="wide")
    update_selectbox_style()
    update_multiselect_style()
    st.markdown("## Demo of Controllable Neural Symbolic Regression")
    st.markdown("""
    Neural Symbolic Regression (NSR) algorithms can quickly identify patterns in data and generate analytical expressions, but lack the capability to incorporate user-defined prior knowledge.

    In this paper we present **Neural Symbolic Regression with Hypothesis** (NSRwH) a novel neural symbolic regression method which enables the explicit incorporation of assumptions about the expected structure of the ground-truth expression into the prediction process.
    
    * Link to the paper: https://arxiv.org/abs/2304.10336
    * Github repository: https://github.com/SymposiumOrganization/SiNesymres

    The following demo allows you to test the NSRwH model and compare it with a standard NSR model.
""")
    
    # Load the picture of the architecture from the assets folder and disply it (is a pdf file)
    show_pdf('assets/main_figure_arxiv.pdf')
     
    st.markdown("### Setup")
    st.markdown("\n Please fill the following fields with the path to the NSRwH and NSR models. Instruction on how to get or \
                train these models can be found in the README.md file")
    nsrwh = st.text_input("Path to the NSRwH model", "ControllableNeuralSymbolicRegressionWeights/nsrwh_200000000_epoch=149.ckpt")
    
    metadata = load_metadata_hdf5(Path("configs"))
    # Retrofit word2id if there is conditioning
    # Set the seeds
    torch.manual_seed(21)
    np.random.seed(21)
    random.seed(21)
    
    beam_size = st.number_input("Select the beam size for the models", 1, 100, 5)

    cfg =  omegaconf.OmegaConf.load(Path("configs/nsrwh_network_config.yaml"))
    cfg.inference.bfgs.activated = True
    cfg.inference.bfgs.n_restarts=10
    cfg.inference.n_jobs=-1
    cfg.dataset.fun_support.max =5
    cfg.dataset.fun_support.min = -5
    cfg.inference.beam_size = beam_size

    metadata = retrofit_word2id(metadata, cfg)
    
    is_cuda = st.checkbox("Tick this if you want to load the models into the GPU", True)
    

    do_inference_with_also_nsr = st.checkbox("Tick this if you want to also run the NSR model", True)
    if do_inference_with_also_nsr:
        nsr = st.text_input("Path to the NSR model", "ControllableNeuralSymbolicRegressionWeights/nsr_200000000_epoch=149.ckpt")
    else:
        nsr = None


    negative_pool =  prepare_negative_pool(cfg) 

    st.markdown("### Define the equation to test") 
    equation_examples = ["0.01*x_1+x_2+exp(x_3)", "sin(x_1)+sqrt(x_2)+sin(x_3+x_4)", "0.5*x_1**(1/2)+x_2**2 + x_3**2", "exp(0.043*sin(x_1*x_2))+x_3", "x_1**2+log(x_3+x_2)", "other"]
    eq_string = st.selectbox("Select an equation or select on 'other' to write your own equation to test", equation_examples, index=4)

    if eq_string == "other":
        eq_string = st.text_input("Enter equation", "x_1**2+x_2**2+x_3**2")
        

    eq_sympy_infix_with_constants = sympify_equation(eq_string)
    eq_sympy_prefix_with_constants = Generator.sympy_to_prefix(eq_sympy_infix_with_constants, enable_float=True)
    costants, eq_sympy_prefix_with_c= return_costants(eq_sympy_prefix_with_constants)
    tmp = list(eq_sympy_infix_with_constants.free_symbols)
    variables = sorted([str(x) for x in tmp])
    f = sympy.lambdify(variables, eq_sympy_infix_with_constants)

    number_of_points = st.number_input("Select the number of points that you would like to be sampled", 10, 1000, 200)
    noise_applied = st.number_input("Select the amount of noise to be applied to the Y", 0.0, 1.0, 0.0)

    # # Sample the points
    # X = np.random.uniform(-10, 10, (number_of_points, len(variables)))
    # X_dict = {}
    # for idx, var in enumerate(variables):
    #     X_dict[var] = torch.tensor(X[:,idx:idx+1]).half()

    # y = f(**X_dict).T + np.random.normal(0, noise_applied, number_of_points)
    # y = y.squeeze()
    cnt = 0
    MAX_ATTEMPTS = 5
    while cnt < MAX_ATTEMPTS:
        support_limits = return_support_limits(cfg, metadata, support=None)
        support = sample_support(support_limits, variables, cfg.dataset.max_number_of_points*5,  metadata.total_variables, cfg)
        is_valid, data_points = sample_images(f, support, variables, cfg)
        if is_valid:
            break
        cnt += 1
    if not is_valid:
        raise ValueError("Could not find a valid support")
    
    # Shuffle the datapoints along the points dimension
    data_points = data_points[:, :, torch.randperm(data_points.shape[2])]
    data_points = data_points[:, :, :number_of_points]
    X = data_points[0,:5,:].T
    y = data_points[0,5,:]
    
    # X, y = get_robust_random_data(eq_string, variables, cfg)
    # pts = torch.arange(number_of_points)
    # pts = torch.randperm(len(pts))
    
    
   

    if is_cuda:
        X = torch.tensor(X).cuda()
        y = torch.tensor(y).cuda()

    # Get all the properties from the equation
    properties = compute_properties(eq_string, compute_symmetry=True,metadata=metadata, cfg=cfg, is_streamlit=True)
    st.markdown("### Select which additional information to pass to NSRwH")
    st.write("""
            "As explained in the paper we defined four different types of conditioning that can be passed to the model. These include complexity, symmetry, appearing branches and appearing constants as well as absent branches:
            * Complexity is defined as the number of mathematical operators, features and constants in the output prediction.
            * Symmetry is defined as the concept of generalized symmetry proposed in [Udrescu et al., 2020]
            * Appearing branches is any branch of the tree that appears in the ground-truth expression (i.e. Positive Conditioning)
            * Absent branches is any branch that does not appear in the ground-truth expression (i.e. Negative Conditioning)
            """
            )
            
    conditioning_to_give = st.multiselect("Select conditionings:", ["Complexity", "Symmetry", "Appearing branches", "Absent branches"], ["Appearing branches"])
    pointer_words = None
    description = {"positive_prefix_examples": [], "negative_prefix_examples": []}
    if "Complexity" in conditioning_to_give:
        st.markdown("###### Complexity")
        gt_complexity = properties["complexity"].split("=")[1]
        complexity = st.number_input(f"Select the target complexity. 1 is the lowest complexity ($$x_1$$) while 20 is the highest. The ground truth has a complexity of {gt_complexity}", 1, 20,int(gt_complexity))
        description["complexity"] = properties["complexity"].split("=")[0]+ "=" + str(complexity)

    if "Symmetry" in conditioning_to_give:
        st.markdown("###### Symmetry")
        if len(properties["symmetry"]) < 2:
            st.write("The ground truth expression does not have any non-trivial symmetry")
        symetries_avaiable = [x.split("=")[1] for x in properties["symmetry"]]
        st.write(f"The ground truth expression has the following symmetries: {symetries_avaiable}, passing them to the model")
        description["symmetry"] = properties["symmetry"]
                
    if "Appearing branches" in conditioning_to_give:
        st.markdown("###### Appearing branches")
        gt_appearing_branches = properties["all_positives_examples"]
        appearing_branches = st.multiselect("Select which appearing branches to pass (Max 2)", gt_appearing_branches, gt_appearing_branches[2:3]+ gt_appearing_branches[10:11])
        assert len(appearing_branches) <= 2, "You can only select up to 2 appearing branches"
        # for branch in appearing_branches:
        #     mix_ptr_constants(branch, cfg)
        constants = set()
        for entry in appearing_branches:
            for xxx in entry:
                if is_token_constant(xxx):
                    constants.add((xxx,))
        appearing_branches = list(constants) + appearing_branches
        # Remove duplicates
        appearing_branches = list(set([tuple(x) for x in appearing_branches]))

        # Sort the appearing branches by length
        appearing_branches = sorted(appearing_branches, key=lambda x: len(x))

            
        positive_symbolic_conditionings, pointer_examples, pointer_to_cost, pointer_words = prepare_pointers(appearing_branches)
        positive_symbolic_conditionings = [x for x in positive_symbolic_conditionings if len(x) > 1 or x[0] not in pointer_words]
        symbolic_conditionings = pointer_examples + positive_symbolic_conditionings

        description["positive_prefix_examples"] = symbolic_conditionings
        description["cost_to_pointer"] = pointer_to_cost
        

    if "Absent branches" in conditioning_to_give:
        st.markdown("###### Absent branches")
        cfg.dataset.conditioning.negative.min_percent = 25
        cfg.dataset.conditioning.negative.max_percent = 25
        cfg.dataset.conditioning.negative.prob = 1
        cfg.dataset.conditioning.negative.k = 500
        cfg.dataset.conditioning.negative.sampling_type = "x^4"
        
        
        negative_candidates = create_negatives(eq_sympy_prefix_with_c, negative_pool, all_positives_examples=properties["all_positives_examples"], metadata=metadata, cfg=cfg)
        good_negative_candidates = []
        for candidate in negative_candidates:
            try:
                tokenize(candidate, metadata.word2id)
            except:
                continue
            good_negative_candidates.append(candidate)
        negative_candidates = good_negative_candidates

        negative_examples = st.multiselect("Select which absent branches to pass (max 2)", negative_candidates, negative_candidates[:2])
        assert len(negative_examples) <= 2, "You can only select up to 2 absent branches"

        # Sort the negative branches by length
        negative_examples = sorted(negative_examples, key=lambda x: len(x))

        description["negative_prefix_examples"] = negative_examples

    # Prepare the conditioning
    cond_tokens, cond_str_tokens = description2tokens(description, metadata.word2id , cfg)

    if is_cuda:
        cond_tokens = torch.tensor(cond_tokens).long().cuda()
    else:
        cond_tokens = torch.tensor(cond_tokens).long()

    if pointer_words is not None:
        numberical_conditioning = [float(description["cost_to_pointer"][key]) for key in pointer_words if key in description["cost_to_pointer"]]
    else:
        numberical_conditioning = []

    if is_cuda:
        conditioning = {"symbolic_conditioning": cond_tokens, "numerical_conditioning": torch.tensor(numberical_conditioning,device="cuda").float()}
    else:
        conditioning = {"symbolic_conditioning": cond_tokens, "numerical_conditioning": torch.tensor(numberical_conditioning,device="cpu").float()}
    #conditioning = {"symbolic_conditioning": torch.tensor([1,2],device="cuda").long(), "numerical_conditioning": torch.tensor([],device="cuda").float()}

    fit = st.button("Run the model")
    if fit:
        if is_cuda:
            fitfunc = return_fitfunc(cfg, metadata, nsrwh, device="cuda")
        else:
            fitfunc = return_fitfunc(cfg, metadata, nsrwh, device="cpu")
            
        new_outputs = fitfunc(X, y,conditioning, cond_str_tokens,  is_batch=False)
        
        st.markdown("### Model Evaluation")
        st.markdown("#### NSRwH")

        best_prediction = new_outputs["best_pred"]

        st.markdown("##### Results:")
        # Use latex to display the equation
        st.latex(f"\\text{{Ground truth: }} {sympy.latex(eq_sympy_infix_with_constants)}")
        st.latex(f"\\text{{Top Prediction: }} {sympy.latex(best_prediction)}")
        
        st.markdown("###### Other candidates:")
        for idx, pred in enumerate(new_outputs["all_preds"]):
            if idx == 0:
                continue
            st.latex(f"\\text{{Prediction {idx+1}: }} {sympy.latex(pred)}")


        if nsr is not None:
            cfg_nsr =  omegaconf.OmegaConf.load(Path("configs/nsr_network_config.yaml"))
            cfg_nsr.inference.bfgs.activated = True
            cfg_nsr.inference.bfgs.n_restarts=10
            cfg_nsr.inference.n_jobs=1
            cfg_nsr.inference.beam_size = beam_size
            
            if is_cuda:
                fitfunc_nsr = return_fitfunc(cfg_nsr, metadata, nsr, device="cuda")
            else:
                fitfunc_nsr = return_fitfunc(cfg_nsr, metadata, nsr, device="cpu")

            st.write("The following table shows the results of the NSR model")
            cond = {'symbolic_conditioning': torch.tensor([[1, 2]]), 'numerical_conditioning': torch.tensor([])}
            new_outputs_nsr = fitfunc_nsr(X, y, cond, is_batch=False)
            
            best_prediction_nsr = new_outputs_nsr["best_pred"]
            st.markdown("##### Results:")
            # Use latex to display the equation
            st.latex(f"\\text{{Ground truth: }} {sympy.latex(eq_sympy_infix_with_constants)}")
            st.latex(f"\\text{{Top Prediction: }} {sympy.latex(best_prediction_nsr)}")
            
            st.markdown("###### Other candidates:")
            for idx, pred in enumerate(new_outputs_nsr["all_preds"]):
                if idx == 0:
                    continue
                st.latex(f"\\text{{Prediction {idx+1}: }} {sympy.latex(pred)}")



 

if __name__ == '__main__':
    main()