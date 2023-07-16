import streamlit as st 
import pandas as pd
from sympy import sympify, latex, print_mathml
from tqdm import tqdm
from pathlib import Path

@st.cache
def prepare_data_with_sympify(df_dict, x):
    # Apply sympify to the raw_pred
    for i in tqdm(range(len(df_dict[x]["raw_pred"]))):
        if type(df_dict[x]["raw_pred"][i]) == str and "constant"  in df_dict[x]["raw_pred"][i]:
            df_dict[x]["raw_pred"][i] = df_dict[x]["raw_pred"][i].format(constant="constant")
        df_dict[x]["raw_pred"][i] = sympify(df_dict[x]["raw_pred"][i])
        #df_dict[x]["pred"][i] = sympify(df_dict[x]["pred"][i])
        df_dict[x]["gt_print"] = df_dict[x]["gt"]
        df_dict[x]["gt_print"][i] = sympify(df_dict[x]["gt"][i])
    return df_dict


def load_data(runs,experiment):
    df_dict = {}
    is_nicer = st.selectbox("Make the entry nicer?", ["Yes", "No"], index=1)
    for x in runs:
        curr = pd.read_pickle(f"luca_results/results_batched_{x}_experiment_type_{experiment}_points.pkl")
        cond_dataset = curr['entry']
        cond_given = [x.split("-")[0] for x in cond_dataset]
        dataset = [x.split("-")[1] for x in cond_dataset]
        batch_number = [int(Path(x).stem.split("_")[-1]) for x in curr['file_name']]
        curr['dataset'] = dataset
        curr['cond_given'] = cond_given
        curr['batch_number'] = batch_number
        df_dict[x] = curr
        # Apply sympify to the raw_pred
        if is_nicer == "Yes":
            df_dict = prepare_data_with_sympify(df_dict.copy(), x)
    return df_dict

def main():
    # Set wide layout
    st.set_page_config(layout="wide")
    
    runs = [str(int(st.number_input("Run", value=54, step=1)))]
    # Select experiment 
    experiment = st.sidebar.selectbox("Experiment Type", ["1", "2"], index=1) 
    df_dict = load_data(runs,experiment)
    df_dict = df_dict.copy()
    df_init = df_dict[runs[0]]
    conditions_available =  list(df_init["cond_given"].unique())
    points_available = list(df_init["number_of_point"].unique())
    noise_levels = list(df_init["noise"].unique())

    dataset = st.sidebar.selectbox("Dataset", df_init['dataset'].unique())
    keep_only_identical = st.sidebar.checkbox("Keep only identical conditions", value=False)
    #keep_only_costs_expr = st.sidebar.checkbox("Keep only expression with constants", value=False)

    epochs = list(df_dict.keys())
    # Select condition1 and 2
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

 
    # Select epoch 
    dfs = []
    for idx, col in enumerate(cols):
        curr = col.selectbox(f"Epoch {idx}", epochs)
        dfs.append(df_dict[str(curr)])
    
    # Set the dataset
    tmp = []
    for idx, col in enumerate(cols):
        curr_df = dfs[idx]
        filtered_df = curr_df[curr_df['dataset'] == dataset]
        tmp.append(filtered_df)
    dfs = tmp


    # Select condition
    tmp = []
    for idx, col in enumerate(cols):
        condition = col.selectbox(f"Condition {idx}", conditions_available)
        curr_df = dfs[idx]
        filtered_df = curr_df[curr_df['cond_given'] == condition]
        tmp.append(filtered_df)
    dfs = tmp

    # Select points 
    tmp = []
    for idx, col in enumerate(cols):
        points = col.selectbox(f"Point {idx}", points_available)
        curr_df = dfs[idx]
        filtered_df = curr_df[curr_df['number_of_point'] == points]
        tmp.append(filtered_df)
    dfs = tmp

    # Select noise level
    tmp = []
    for idx, col in enumerate(cols):
        noise = col.selectbox(f"Noise {idx}", noise_levels)
        curr_df = dfs[idx]
        filtered_df = curr_df[curr_df['noise'] == noise]
        tmp.append(filtered_df)
    dfs = tmp

    tmp = []
    for idx, df in enumerate(dfs):
        sorted_df = df.sort_values(by=['idx'])
        tmp.append(sorted_df)
    dfs = tmp
    
    number_of_hypothesis = len(dfs[0]["cond_positive"].iloc[0])
    number_chosen = st.number_input("Number of candidates", min_value=1, max_value=number_of_hypothesis, value=number_of_hypothesis)



    for col,df in zip(cols, dfs):
        #if "ood_max_pointwise_acc" in df.columns:
        ood_pt_acc = df["ood_pointwise_acc"] >= 0.99
        ood_r2_acc = df["ood_r2"] > 0.999
        iid_pt_acc = df["iid_pointwise_acc"] >= 0.99  
        iid_r2_acc = df["iid_r2"] > 0.999

        symmetry =  df["cond_symmetry"]
        complexity = df["cond_complexity"]
        positive = df["cond_positive"]
        negative = df["cond_negative"]

        # col.write("Mean r2>0.99: " + str(r99.mean()))
        # col.write("Mean r2>0.999: " + str(r999_1.mean()))
        # col.write("Mean PointAcc 99: " + str(pt_acc.mean()))
        # col.write("Mean r2-mean: " + str(rmedian.mean()))
        col.write("OOD Point Accuracy: " + str(ood_pt_acc.mean()))
        col.write("OOD R2 Accuracy: " + str(ood_r2_acc.mean()))
        col.write("IID Point Accuracy: " + str(iid_pt_acc.mean()))
        col.write("IID R2 Accuracy: " + str(iid_r2_acc.mean()))


        conds = ["cond_symmetry", "cond_complexity", "cond_positive", "cond_negative"]
        for cond in conds:
            df[cond] = df[cond].apply(lambda x: sum(x[:number_chosen])/number_chosen)
            col.write("Mean " + cond + ": " + str(df[cond].mean()))
        
        # col.write("Mean symmetry: " + str(symmetry.mean()))
        # col.write("Mean complexity: " + str(complexity.mean()))
        # col.write("Mean positive: " + str(positive.mean()))
        # col.write("Mean negative: " + str(negative.mean()))

    # # Select condition1 and 2
    # col0, col1, col2, col3, col4 = st.columns(5)
    # gt = df1.sort_values(by=['gt'])[["gt"]]

    # col0.write(gt)
    # # Merge dataframes
    # df = pd.merge(df1, df2, on='gt', suffixes=('_1', '_2'))
    # df = pd.merge(df, df3, on='gt', suffixes=('', '_3'))
    # df = pd.merge(df, df4, on='gt', suffixes=('', '_4'))

    # Set the idx column as index
    cnt = 0

    if st.button("Save csv"):
        save = True
    else:
        save = False
    for col,df in zip(cols, dfs):
        # Reorder dataframe as the gt column, including duplicates
        df = df.sort_values(by=['batch_number', 'idx'])
        df = df.reset_index(drop=True)
        to_display_df = df[["gt","best_pred","best_raw_pred","condition_str_tokenized","cond_given","cond_symmetry","cond_complexity","cond_positive","cond_negative","ood_pointwise_acc","ood_r2","iid_pointwise_acc","iid_r2","bfgs_loss"]]
        
        # to_display_df["raw_pred"] = to_display_df["raw_pred"].apply(lambda x: str((x)))
        # # to_display_df["raw_pred"] = to_display_df["raw_pred"].apply(lambda x: f"${x}$")
        # col.write(to_display_df, unsafe_allow_html=True)
        st.write(to_display_df, unsafe_allow_html=True)

        # Create a button to save the csv
        if save:
            to_display_df.to_csv(f"data_{cnt}.csv", index=False)
            st.write("Saved csv file")

        cnt += 1
    

if __name__ == "__main__":
    main()