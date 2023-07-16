import pandas as pd 

def main():
    #df_cat = pd.read_pickle('result_blackbox_vTue_night.pkl')
    df_succ = pd.read_pickle('result_blackbox_vwed_night.pkl')
    df_succ_2 = pd.read_pickle('result_blackbox_vwed_night_v4.pkl')
    df = pd.concat([df_succ, df_succ_2], axis=0)
    #gomea_csv = pd.read_csv("result_blackbox.csv")
    # Set values at column 0 as index
    #df = df.set_index(df.iloc[:, 0])
    #df = df.drop(df.columns[0], axis=1)
    #df = df.iloc[:12]
    # Get all columns from the third to the fourth last
    #df = df.iloc[:, 3:-4]
    df.iloc[:, 3:-4] = df.iloc[:, 3:-4].astype(float)
    #df = df.T

    
    eqs = df["idx_eq"].unique()
    total_r2_candidates = []
    total_r2_vanillas = []
    total_pointwise_candidates = []
    total_pointwise_vanillas = []
    seeds_available = df["seed"].unique()
    total_r2_gomea = []
    is_correct = []

    to_be_save_new =  df.iloc[:,:-3]
    to_be_save_new["division"] = to_be_save_new["read_sentence"] == 'None'
    to_check = to_be_save_new[(to_be_save_new["pointwise"] > 0.99) & (to_be_save_new["eq_name"] == "strogatz_glider2")]
    to_be_save_ready = to_be_save_new[["seed","eq_name","idx_eq","dim_eq","idx","r2_test","r2_train","r2_val", "err_val", "err_train", "pointwise","division"]]                                        
    to_be_save_ready.to_pickle("results_black_box_plot_thursday.pkl")

    for seed in seeds_available[:]:
        for eq in eqs:
            df_eq = df[df["idx_eq"] == eq]
            df_eq = df_eq[df_eq["seed"] == seed]
            
            try:
                eq_name = df_eq["eq_name"].iloc[0]
            except:
                break
            # if not "strogatz" in eq_name:
            #     continue
                #gomea_csv_eq = gomea_csv[gomea_csv["dataset"] == eq_name]
                #gomea_csv_eq = gomea_csv_eq[gomea_csv_eq["random_state"] == int(chosen_seed)]['r2_test']
            # Find the argmin of err_train
            # Get the entries with None in read_sentence
            df_eq_None = df_eq[df_eq["read_sentence"] == 'None']
            df_eq_not_None = df_eq[df_eq["read_sentence"] != 'None']
            print(len(df_eq_not_None))
            idx_err_min = pd.to_numeric(df_eq_None["r2_val"]).argmax() 
            vanilla = df_eq_None.iloc[idx_err_min]

            try:
                idx_err_min = pd.to_numeric(df_eq_not_None["r2_val"]).argmax() 
                
            except:
                break

            df_candidate = df_eq_not_None.iloc[idx_err_min]

            print("Dataset:", df_eq["eq_name"].iloc[0])
            print("R2 Candidate: ", df_candidate["r2_test"])
            # if len(gomea_csv_eq) > 0:
            #     print("R2 Gomea: ", gomea_csv_eq.iloc[0])
            total_r2_candidates += [df_candidate["r2_test"]]

            # if len(gomea_csv_eq) > 0:
            #     total_r2_gomea += [gomea_csv_eq.iloc[0]]
            print("R2 Vanilla:", vanilla["r2_test"])
            total_r2_vanillas += [vanilla["r2_test"]]
            print("PointWise Candidate:", df_candidate["pointwise"])
            total_pointwise_candidates += [df_candidate["pointwise"]]
            print("PointWise Vanilla:", vanilla["pointwise"])
            total_pointwise_vanillas += [vanilla["pointwise"]]


    # Compute the median of the r2 in both cases
    first = pd.DataFrame(total_r2_candidates).median()
    second = pd.DataFrame(total_r2_vanillas).median()
    # Replace negative values by 0
    total_r2_candidates = [0 if x < 0 else x for x in total_r2_candidates]
    total_r2_vanillas = [0 if x < 0 else x for x in total_r2_vanillas]
    first_mean = pd.DataFrame(total_r2_candidates).mean()
    second_mean = pd.DataFrame(total_r2_vanillas).mean()

    import numpy as np

    are_higher_than099_candidates = np.mean([x > 0.99 for x in total_r2_candidates])
    are_higher_than099_vanillas = np.mean([x > 0.99 for x in total_r2_vanillas])

    is_correct_candidates = np.mean([x > 0.99 for x in total_pointwise_candidates])
    is_correct_vanillas = np.mean([x > 0.99 for x in total_pointwise_vanillas])


    #third = pd.DataFrame(total_r2_gomea).median()
    print("Median R2 Candidate:", first)
    print("Median R2 Vanilla:", second)
    print("Mean R2 Candidate:", first_mean)
    print("Mean R2 Vanilla:", second_mean)
    print("Percentage of candidates higher than 0.99:", are_higher_than099_candidates)
    print("Percentage of vanillas higher than 0.99:", are_higher_than099_vanillas)
    print("Percentage of candidates correct:", is_correct_candidates)
    print("Percentage of vanillas correct:", is_correct_vanillas)
    #print("Median R2 Gomea:", third)

    # first = pd.DataFrame(total_pointwise_candidates).median()
    # second = pd.DataFrame(total_pointwise_vanillas).median()
    # print("Median PointWise Candidate:", first)
    # print("Median PointWise Vanilla:", second)
    # eqs = df
    # # Find the argmin of err_train

    # breakpoint()
    # print(df.head())

if __name__ == "__main__":
    main()