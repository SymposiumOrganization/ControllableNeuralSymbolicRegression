import pandas as pd
from sympy import sympify, latex, print_mathml
from tqdm import tqdm
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from collections import defaultdict
import itertools 

name_to_symbol_dict = {
    'vanilla': 'o',
    'positive': 'v',
    'standard_nesy': '>',
    'all': '*',
    'symmetry': 's',
    'complexity': 'd',
    'negative': 'p',
    'constants': 'h'
}
colors = sns.color_palette("Set2")
name_to_color_dict = {
    'vanilla': colors[0],
    'positive': colors[1],
    'standard_nesy': colors[2],
    'all': colors[3],
    'symmetry': colors[4],
    'complexity': colors[5],
    'negative': colors[6],
    'constants': colors[7]
}

standard_order =   ['all', 'complexity', 'constants', 'negative', 'positive', 'standard_nesy', 'symmetry']    
standard_vs_vanilla = ['standard_nesy', 'vanilla']

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

def load_data(run,experiment, keywords=None):
    df_dict = {}
    if keywords is None:
        curr = pd.read_pickle(f"luca_results/results_batched_{run}_experiment_type_{experiment}.pkl")
    else:
        curr = pd.read_pickle(f"luca_results/results_batched_{run}_experiment_type_{experiment}_{keywords}.pkl")
    cond_dataset = curr['entry']
    if experiment==str(3):
        cond_given = cond_dataset
        dataset = []
        import numpy as np
        for xx in np.array(curr["file_name"]):
            if 'only_five_variables_wc' in xx:
                temp = 'only_five_variables_wc'
            elif 'train_wc' in xx :
                temp = 'train_wc'
            else:
                raise KeyError()
            dataset.append(temp)
        batch_number = [0 for x in range(len(cond_dataset))]
    else:
        cond_given = [x.split("-")[0] for x in cond_dataset]
        dataset = [x.split("-")[1] for x in cond_dataset]
        batch_number = [int(Path(x).stem.split("_")[-1]) for x in curr['file_name']]
    # cond_dataset = curr['entry']
    # cond_given = [x.split("-")[0] for x in cond_dataset]
    # dataset = [x.split("-")[1] for x in cond_dataset]
    batch_number = [int(Path(x).stem.split("_")[-1]) for x in curr['file_name']]
    curr['dataset'] = dataset
    curr['cond_given'] = cond_given
    curr['batch_number'] = batch_number
    # Apply sympify to the raw_pred
    return curr

# Select experiment 

    


def extract(df, dataset,condition,points,noise,number_chosen):
    df = df[df['dataset'] == dataset]
    df = df[df['cond_given'] == condition]
    df = df[df['number_of_point'] == points]
    df = df[df['noise'] == noise]
    df = df.sort_values(by=['idx'])    
    conds = ["cond_symmetry", "cond_complexity", "cond_positive", "cond_negative"]
    for cond in conds:
        df[cond] = df[cond].apply(lambda x: sum(x[:number_chosen])/number_chosen)

    return df

def plot_controllability_appendix(raw_df):
    datasets= ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
    metrics = ['cond_positive','cond_complexity','cond_negative','cond_symmetry']
    points = 400
    noises = [0.0,0.1]
    colors = sns.color_palette("Set2")
    sizes = [2]
    dataset_ticks= ['train_nc','train_wc','ofv_nc','aif']
    m_shape = ['o','v','>','*']

    for noise in noises:
        for metric in tqdm(metrics):
            if 'positive' in metric:
                labels_cond = ['vanilla','positive','standard_nesy','all']
            if 'negative' in metric:
                labels_cond = ['vanilla','negative','standard_nesy','all']
            if 'complexity' in metric:
                labels_cond = ['vanilla','complexity','standard_nesy','all']
            if 'symmetry' in metric:
                labels_cond = ['vanilla','symmetry','standard_nesy','all']

            fig, host = plt.subplots(figsize=(8,5)) # (width, height) in inches

            x0 = [0.5]*6
            x1 = [1.5]*6
            x2 = [2.5]*6
            x3 = [3.5]*6
            xs = [x0,x1,x2,x3]
            
            trajs =  [ [] for _ in range(len(datasets))]
            for j,lab in enumerate(labels_cond):
                
                for idx, dataset in enumerate(datasets):
                    df = extract(raw_df,dataset,lab,points,noise,sizes[0])
                    curr = df[metric].mean()
                    trajs[idx].append(curr)

            parts = [host, host.twinx(), host.twinx(), host.twinx()]
            size_m = 12

            min_max_dict = [{"min":1000,"max":0} for i in range(4)]
            
            for j in range(4): # Label Conditions
                points = []
                xs_j = [x[j] for x in xs]
               
                current_trajs = [traj[j] for traj in trajs]

                for idx_dataset, traj in enumerate(current_trajs): # Datasets
                    #curr_max = min_max_dict[idx_traj]["max"]
                    part = parts[idx_dataset]
                    #curr_min = min_max_dict[idx_traj]["min"]
                    #curr_element_traj = current_trajs[idx_traj]
                    curr_point, =part.plot(xs_j[idx_dataset],traj,m_shape[j],color=colors[j],markersize=size_m,markeredgewidth=1, markeredgecolor='k')
                    points.append(curr_point)
                    if traj<min_max_dict[idx_dataset]["min"]:
                        min_max_dict[idx_dataset]["min"]= traj
                    if traj>min_max_dict[idx_dataset]["max"]:
                        min_max_dict[idx_dataset]["max"] = traj

            for i in range(1,4):
                curr_min = min_max_dict[i]["min"]
                curr_max = min_max_dict[i]["max"]
                parts[i].set_ylim(curr_min-0.15, curr_max+0.15)

            parts[0].set_xlim(0, 4)
            parts[0].tick_params(axis="y",direction="in", pad=-32)
            parts[1].spines['right'].set_position(('outward', -360))
            parts[2].spines['right'].set_position(('outward', -240))
            parts[3].spines['right'].set_position(('outward', -130))

            # no x-ticks
            xs0 = [x[0] for x in xs]
            host.xaxis.set_ticks(xs0)    
            host.set_xticklabels(dataset_ticks)             
            #par2.xaxis.set_ticks([])
            #par3.xaxis.set_ticks([])

            for i in range(4):
                parts[i].yaxis.label.set_color(points[i].get_color())
            # host.yaxis.label.set_color(p1.get_color())
            # par1.yaxis.label.set_color(p2.get_color())
            # par2.yaxis.label.set_color(p3.get_color())
            # par3.yaxis.label.set_color(p4.get_color())


            legend_elements = [mlines.Line2D([], [], linestyle='None', marker=m_shape[0], markersize=size_m,label = labels_cond[0], markeredgewidth=1, markeredgecolor='k'),
                            mlines.Line2D([], [], linestyle='None', marker=m_shape[1], markersize=size_m,label = labels_cond[1], markeredgewidth=1, markeredgecolor='k'),
                            mlines.Line2D([], [], linestyle='None', marker=m_shape[2], markersize=size_m,label = labels_cond[2], markeredgewidth=1, markeredgecolor='k'),
                            mlines.Line2D([], [], linestyle='None', marker=m_shape[3], markersize=size_m,label = labels_cond[3], markeredgewidth=1, markeredgecolor='k')]

            # Create the figure
            host.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.05),ncol=6, fontsize=12)

            host.set_ylabel(metric,fontsize=12,color='k')

            # Adjust spacings w.r.t. figsize
            fig.tight_layout()
            # Alternatively: bbox_inches='tight' within the plt.savefig function 
            #                (overwrites figsize)
            plt.savefig('63b2f4a9f72eb16649dabe85/figs/appendix_controllability1_{}_{}_{}.png'.format(metric,noise,sizes[0]))

def plot_controllability_main_body():
    raw_df = load_data('149',str(1))
    metrics = ['cond_positive','cond_complexity','cond_negative','cond_symmetry']
    points = 400
    noises = [0.0,0.1]
    colors = sns.color_palette("Set2")
    sizes = [1,16,32,64,128,256]
    datasets = ["average"]
    for metric in metrics:
        if 'positive' in metric:
            labels_cond = ['positive','all','vanilla']
            iter_datasets= ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
        if 'negative' in metric:
            labels_cond = ['negative','all','vanilla']
            iter_datasets= ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
        if 'complexity' in metric:
            labels_cond = ['complexity','all','vanilla']
            iter_datasets= ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
        if 'symmetry' in metric:
            labels_cond = ['symmetry','all','vanilla']
            iter_datasets= ['only_five_variables_nc', 'only_five_variables_wc']

        for dataset in datasets:
            
            plt.figure()
            #plt.title(dataset + ' ' + metric)
            plt.xlabel(r"Beam Size",fontsize=17)
            metric_name = metric.split("cond_")[1]
            # Make first letter capital
            metric_name = metric_name[0].upper() + metric_name[1:]

            plt.ylabel('{} Is_satisfied (%)'.format(metric_name),fontsize=17)

            # Create cartesian product of noise and labels conditions
            label_to_idx = {label: idx for idx, label in enumerate(labels_cond)}
            comb_noise = list(itertools.product(labels_cond,noises ))
            for entry in comb_noise:
                lab, noise  = entry
                ticks_labels = sizes
                #m_shape = ['o','v','>','*']

                #for _,lab in enumerate(labels_cond):
                x = []
                y = []
                y_std = []
                for i,p in tqdm(enumerate(sizes)):
                    if dataset == 'average':
                        values = []
                        for sub in iter_datasets:
                            dataframe = extract(raw_df, sub,lab,points,noise,p)
                            #if  'complexity' in metric:
                                #complexity = 1 - (dataframe[metric])/20
                            curr = dataframe[metric].mean()
                            if  'complexity' in metric:
                                curr = 1 - curr/20
                            values.append(curr)
                    traj = np.nanmean(values)
                    traj_std = np.nanstd(values)
                    x.append(i)
                    y.append(traj)
                    y_std.append(traj_std)

                # Depending on the noise level change the color brightness and huee
                j = label_to_idx[lab]
                if noise == 0.0:
                    colors[j] = sns.color_palette("Set2")[j]
                else:
                    color_candidate = sns.color_palette("Set2")[j]
                    # Change lower brightness
                    colors[j] = [color_candidate[0],color_candidate[1],color_candidate[2]+ 0.2]


                # Add X, y 
                shape = name_to_symbol_dict[lab]
                color = name_to_color_dict[lab]
                # if noise == 0.0:
                #     entry = f"{lab}"
                # else:
                #     entry = f"{lab} noisy"
                if lab == 'vanilla':
                    name = "standard_nesy"
                    color = name_to_color_dict["standard_nesy"]
                    shape = name_to_symbol_dict["standard_nesy"]
                else:
                    name = lab
                plt.plot(x, y,shape, markersize=10, label=name, color = color,markeredgewidth=1, markeredgecolor='k')
                # Add standard deviation
                #plt.fill_between(x, np.array(y) - np.array(y_std), np.array(y) + np.array(y_std), alpha=0.2, color = colors[j])
                #plt.plot(x, y,'--', color = colors[j])
                if noise != 0.0:
                    plt.plot(x, y,'--', color = color)
                else:
                    plt.plot(x, y, color = color)

                        
            plt.xticks(x,ticks_labels)
            # Remove the legend from the plot
            plt.legend().set_visible(True)
            # Remove from legend repeated entries
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            

            
            # Make legend smaller

            if metric_name == "Negative":
                # Move the legend inside inside the axis but on the right
                plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.72, 0.7), loc='center left', borderaxespad=0, fontsize=9)
            else:
                plt.legend(by_label.values(), by_label.keys(), fontsize=9)
            # if metric_name == "Positive":  
            #     plt.legend(bbox_to_anchor=(0.3, 0.4), loc='center', borderaxespad=0, fontsize=9)
            plt.savefig('63b2f4a9f72eb16649dabe85/figs/main_controllability1_{}_all_{}.pdf'.format(metric,noise,sizes[0]), bbox_inches='tight')
            plt.savefig('63b2f4a9f72eb16649dabe85/figs/main_controllability1_{}_all_{}.png'.format(metric,noise,sizes[0]), bbox_inches='tight')
            
            #if metric_name 
            #if metric_name == "Negative":
                # Move the legend to the right 

            

           

def plot_heatmap():
    raw_df = load_data('149',str(3))
    is_correct = raw_df["ood_pointwise_acc"] >= 0.99
    raw_df["is_correct"] = is_correct

    # Dataset only_five_variables_wc
    curr_df = raw_df[raw_df["dataset"] == "only_five_variables_wc"]
    curr_df = raw_df[raw_df["dataset"] == "train_wc"]

    res = {}

    # Group by the cond_given 
    for cond, group in curr_df.groupby("cond_given"):
        # Compute the mean of the is_correct column
        print("Len group {}".format(len(group)))
        mean = group["is_correct"].iloc[:300].mean()
        print("The mean accuracy for cond_given {} is {:.2f}".format(cond, mean))
        res = {**res, **{cond: mean}}
    
    tmp = pd.DataFrame.from_dict(res, orient='index')
    constant_columns = ["0_constant","20_constant","40_constant","60_constant","80_constant","100_constant"]
    positive_row = ["0_positive","20_positive","40_positive","60_positive","80_positive","100_positive"]
    matrix_df = pd.DataFrame(columns=constant_columns, index=positive_row)
    for col in constant_columns:
        for row in positive_row:
            entry = col + "_" + row
            matrix_df.loc[row,col] = float(tmp.loc[entry].iloc[0])
        
    matrix_array = matrix_df.to_numpy().astype(np.float).T

    # Change the aspect ratio to be a rectangle with a width of 2 and height of 1


    # Interpolate rows so that they are only 3 rows rather than 6
    matrix_array = np.array([matrix_array[0], (matrix_array[2]+matrix_array[3])/2, matrix_array[5]])

    # Generate a heatmap from matrix_df with a font size of 11

    # Choose more anonymous continuous color map (not cubehelix)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    # Change aspect ratio to be a rectangle with a width of 2 and height of 1
    plt.figure(figsize=(8, 4))


    fig = sns.heatmap(matrix_array, annot=False, fmt=".2f", cmap=cmap, square=True, cbar = True)
    # Make the cbar shorter
    cbar = fig.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    

    # Change the x and y labels
    plt.ylabel(r"Constants probability %",fontsize=17)
    plt.xlabel(r"Norm. conditioning length",fontsize=17)

    # Change the x and y ticks from 0 to 100
    # Make the ticks in the middle of the squares
    plt.yticks(np.arange(0.5, 3.5, 1), ["0", "50", "100"])
    plt.xticks(np.arange(0.5, 6, 1,), ["0", "0.2", "0.4", "0.6", "0.8", "1.0" ])




    # Save the heatmap to a file
    plt.savefig("63b2f4a9f72eb16649dabe85/figs/heatmap.png", bbox_inches="tight")
    plt.savefig("63b2f4a9f72eb16649dabe85/figs/heatmap.pdf", bbox_inches="tight")


def plot_main_accuracy_presentation(metric=None, is_vanilla_vs_nesy=False):
    raw_df = load_data('149',str(2), keywords="points")
    datasets= ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
    metrics = [metric]
    colors = sns.color_palette("Set2")
    sizes = [2]
    dataset_ticks= ['train_nc','train_wc','ofv_nc','aif']
    if is_vanilla_vs_nesy:
        labels_cond = standard_vs_vanilla
    else:
        labels_cond = standard_order 

    for metric in tqdm(metrics):

        fig, host = plt.subplots(figsize=(8,5)) # (width, height) in inches

        
        x0 = [0.5]*8
        x1 = [1.5]*8
        x2 = [2.5]*8
        x3 = [3.5]*8


        xs = [x0,x1,x2,x3]
        
        trajs =  [ [] for _ in range(len(datasets))]
        for j,lab in enumerate(labels_cond):
            for idx, dataset in enumerate(datasets):
                df = extract(raw_df,dataset,lab,400,0,sizes[0])
                df["is_correct"] = df[metric] >= 0.99
                curr = df["is_correct"].mean()
                trajs[idx].append(curr)

        parts = [host, host.twinx(), host.twinx(), host.twinx()]
        size_m = 12

        min_max_dict = [{"min":1000,"max":0} for i in range(4)]
        
        for j, label in enumerate(labels_cond): # Label Conditions
            points = []
            xs_j = [x[j] for x in xs]
            
            current_trajs = [traj[j] for traj in trajs]

            for idx_dataset, traj in enumerate(current_trajs): # Datasets
                #curr_max = min_max_dict[idx_traj]["max"]
                part = parts[idx_dataset]
                #curr_min = min_max_dict[idx_traj]["min"]
                #curr_element_traj = current_trajs[idx_traj]
                shape = name_to_symbol_dict[label]
                color = name_to_color_dict[label]
                curr_point, =part.plot(xs_j[idx_dataset],traj,shape,color=color,markersize=size_m,markeredgewidth=1, markeredgecolor='k')
                points.append(curr_point)
                if traj<min_max_dict[idx_dataset]["min"]:
                    min_max_dict[idx_dataset]["min"]= np.round(traj,2)
                if traj>min_max_dict[idx_dataset]["max"]:
                    min_max_dict[idx_dataset]["max"] = np.round(traj,2)

        for i in range(4):
            curr_min = min_max_dict[i]["min"]
            curr_max = min_max_dict[i]["max"]
            # parts[i].set_ylim(curr_min-0.01, curr_max+0.01)
            # continue
            if metric == "ood_pointwise_acc":
                if i == 1:
                    parts[i].set_ylim(curr_min-0.03, curr_max+0.01)
                elif i == 3:
                    parts[i].set_ylim(curr_min-0.01, curr_max+0.01)
                else:
                    parts[i].set_ylim(curr_min-0.02, curr_max+0.02)
            else:
                if i == 2:
                    parts[i].set_ylim(curr_min-0.01, curr_max+0.03)
                elif i == 1 or i == 3:
                    parts[i].set_ylim(curr_min-0.02, curr_max+0.015)
                else:
                    parts[i].set_ylim(curr_min-0.02, curr_max+0.02)
        

        parts[0].set_xlim(0, 4)
        parts[0].tick_params(axis="y",direction="in", pad=-30)
        # Round ticks to 1 decimal place
        from matplotlib.ticker import FormatStrFormatter
        
        parts[1].spines['right'].set_position(('outward', -64*6))
        parts[2].spines['right'].set_position(('outward', -64*4))
        parts[3].spines['right'].set_position(('outward', -64*2))

        for i in range(4):
            parts[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # no x-ticks
        xs0 = [x[0] for x in xs]
        host.xaxis.set_ticks(xs0)    
        host.set_xticklabels(dataset_ticks)             
        #par2.xaxis.set_ticks([])
        #par3.xaxis.set_ticks([])

        for i in range(4):
            parts[i].yaxis.label.set_color(points[i].get_color())
        # host.yaxis.label.set_color(p1.get_color())
        # par1.yaxis.label.set_color(p2.get_color())
        # par2.yaxis.label.set_color(p3.get_color())
        # par3.yaxis.label.set_color(p4.get_color())

        # # Create a legend for the first line.

        legend_elements = [Line2D([0], [0], marker=name_to_symbol_dict[label], color='w', label=label,
                             markerfacecolor=name_to_color_dict[label], markersize=10, markeredgecolor='k') for i, label in enumerate(labels_cond)]
        
        # Make the symbol and the text closer to each other in the legend
        # and make the legend a bit transparent



        
        # Reduce the fontsize of the tick labels
        host.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=12)

        # Add X axis label  
        if metric == "ood_pointwise_acc":
            host.set_ylabel("Is Correct", fontsize=14,color='k')
        else:
            host.set_ylabel(r"$R^2_{99}$", fontsize=14, color='k')
        #host.set_ylabel("Dataset", fontsize=14)


        # Adjust spacings w.r.t. figsize
        fig.tight_layout()
        # Alternatively: bbox_inches='tight' within the plt.savefig function 
        #                (overwrites figsize)
        plt.savefig(f'63b2f4a9f72eb16649dabe85/figs/main_accuracy_metric_{metric}_is_vanilla_{is_vanilla_vs_nesy}.png')
        plt.savefig(f'63b2f4a9f72eb16649dabe85/figs/main_accuracy_metric_{metric}_is_vanilla_{is_vanilla_vs_nesy}.pdf')

def plot_main_accuracy_points_noise(is_what,metric,is_vanilla_vs_nesy):
    #colors = sns.color_palette("Set2")
    sizes = [2]
    #m_shape = ['o','v','>','*', 's', 'd', 'p', 'h']
    labels_cond =  ['vanilla','positive','standard_nesy','all', 'symmetry', 'complexity', 'negative','constants']

    
    datasets= ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']


    raw_df = load_data('149',str(2), keywords=is_what)
    metrics = [metric]
    if is_vanilla_vs_nesy:
        labels_cond = standard_vs_vanilla #= ["standard_nesy", "vanilla"]
    else:
        labels_cond = standard_order #=  ['all', 'complexity', 'constants', 'negative', 'positive', 'standard_nesy', 'symmetry']
    # if metric == "ood_pointwise_acc":
    #     raw_df = pd.read_pickle("")
    # elif metric == "ood_r2":
    #     raw_df = pd.read_pickle("")
    if is_what == "points":
        iter_things = ['10','25','50','100','200','400']
    elif is_what == "noise":
        noise = sorted([0.001,0,0.0001,0.01,0.1,1])
        # Convert to string
        iter_things = [str(x) for x in noise]
    else:
        raise KeyError

    # Add X, y 
    
    for dataset in tqdm(datasets):
        
        #labels_cond =  ['vanilla','positive','standard_nesy','all', 'symmetry', 'complexity', 'negative','constants']
        for metric in metrics:
            plt.figure()
            if is_what == "points":
                plt.xlabel(r"# Input points",fontsize=12)
            elif is_what == "noise":
                plt.xlabel("# Noise level",fontsize=12)
            if metric == "ood_pointwise_acc":
                plt.ylabel('Is Correct',fontsize=12)
            else:
                # Write R2_99 in latex format
                plt.ylabel(r'$R^2_{99}$',fontsize=12)
                #plt.ylabel('R2_99',fontsize=12)
            x = [0,1,2,3,4,5]
            plt.xticks(x,iter_things)
            for j,lab in enumerate(labels_cond):
                shape = name_to_symbol_dict[lab]
                color = name_to_color_dict[lab]
                x = []
                y = []

                for i, iter_thing in enumerate(iter_things):
                    if is_what == "points":
                        df = extract(raw_df,dataset,lab,float(iter_thing),0,sizes[0])
                    elif is_what == "noise":
                        if iter_thing == "0" and dataset == "train_wc":
                            raw_df_tmp = load_data('149',str(2), keywords="points")
                            df = extract(raw_df_tmp,dataset,lab,400,0,sizes[0])
                        else:
                            df = extract(raw_df,dataset,lab,400,float(iter_thing),sizes[0])
                    # if is_what == "noise":
                    #     breakpoint()
                    df["is_correct"] = df[metric] >= 0.99
                    curr = df["is_correct"].mean()
                    x.append(i)
                    y.append(curr*100)
                plt.plot(x, y,shape, markersize=10, label=lab, color = color,markeredgewidth=1, markeredgecolor='k')
                plt.plot(x, y,'--', color = color)
            
            # Make legend font smaller and the legend in two columns
            plt.legend(fontsize=8.5,  ncol=2)
            plt.savefig(f'63b2f4a9f72eb16649dabe85/figs/main_accuracy_{is_what}_{dataset}_{metric}_is_vanilla_{is_vanilla_vs_nesy}.png', bbox_inches='tight')
            plt.savefig(f'63b2f4a9f72eb16649dabe85/figs/main_accuracy_{is_what}_{dataset}_{metric}_is_vanilla_{is_vanilla_vs_nesy}.pdf', bbox_inches='tight')

# def appendix_vanilla_vs_standard_():
#     datasets= ['average'] #['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
#     datasets = ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
#     metrics = ["ood_pointwise_acc"]

#     points = ['10','25','50','100','200','400']
#     for dataset in tqdm(datasets):
        
#         #labels_cond =  ['positive','standard_nesy','all', 'symmetry', 'complexity', 'negative','constants']

#         labels_cond = ['vanilla','standard_nesy']
#         for metric in metrics:
#             plt.figure()
#             plt.xlabel(r"# Input points",fontsize=12)
#             plt.ylabel('Accuracy (%)',fontsize=12)
#             x = [0,1,2,3,4,5]
#             plt.xticks(x,points)
#             for j,lab in enumerate(labels_cond):
#                 x = []
#                 y = []

#                 for i, point in enumerate(points):
#                     if dataset == 'average':
#                         ys = []
#                         for d in ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']:
#                             df = extract(raw_df,d,lab,int(point),0,sizes[0])
#                             df["is_correct"] = df[metric] > 0.999
#                             curr = df["is_correct"].mean()
#                             ys.append(curr*100)
#                         x.append(i)
                            
#                         y.append(np.nanmean(ys))
#                     else:
#                         df = extract(raw_df,dataset,lab,int(point),0,sizes[0])
#                         df["is_correct"] = df[metric] > 0.999
#                         curr = df["is_correct"].mean()
#                         x.append(i)
#                         y.append(curr*100)
#                 plt.plot(x, y,m_shape[j], markersize=10, label=lab, color = colors[j],markeredgewidth=1, markeredgecolor='k')
#                 plt.plot(x, y,'--', color = colors[j])
            
#             # Make legend font smaller and the legend in two columns
#             plt.legend(fontsize=8.5,  ncol=2)
#             plt.savefig('63b2f4a9f72eb16649dabe85/figs/main_points_{}_{}.png'.format(dataset,"".join(labels_cond)), bbox_inches='tight')
#             plt.savefig('63b2f4a9f72eb16649dabe85/figs/main_points_{}_{}.pdf'.format(dataset,"".join(labels_cond)), bbox_inches='tight')


def table_third_exp():
    import matplotlib.pyplot as plt
    # Generate heat map with matplotlib and seaborn from the dictionary
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 17})
    
    datasets= ['train_nc','train_wc','only_five_variables_nc','aifeymann_processed']
    sizes = [2]
    noises = [0.0,0.1]
    m_shape = ['o','v','>','*', 's', 'd', 'p', 'h']
    labels_cond =  ['vanilla','positive','standard_nesy','all', 'symmetry', 'complexity', 'negative','constants']

    #raw_df = load_data('149',str(2), keywords="points_v2")
    df = pd.read_pickle("results_black_box_plot_thursday.pkl")
    seeds_available = df.loc[df["eq_name"] == "strogatz_vdp2"]["seed"].unique()

    little_df = pd.read_pickle("result_only_small_black_box.pkl")
    little_seed_available = little_df.loc[little_df["eq_name"] == "strogatz_vdp2"]["seed"].unique()
    assert set(seeds_available).issubset(set(little_seed_available))
    
    eqs = df["idx_eq"].unique()
    total_r2_candidates, total_r2_vanillas,total_r2_little, total_pointwise_candidates, total_pointwise_vanillas, total_poinwise_little = [], [], [], [], [], []
    seeds = []
    eq_name = []
    for seed in seeds_available:
        for eq in eqs:
            df_eq = df[df["idx_eq"] == eq]
            df_eq = df_eq[df_eq["seed"] == seed]
            df_eq_None = df_eq[df_eq["division"] == True]
            df_eq_not_None = df_eq[df_eq["division"] == False]
            little_df_eq = little_df[(little_df["idx_eq"] == eq) & (little_df["seed"] == seed)]
            
            idx_err_min = pd.to_numeric(df_eq_None["r2_val"]).argmax() 
            vanilla = df_eq_None.iloc[idx_err_min]
            idx_err_min = pd.to_numeric(df_eq_not_None["r2_val"]).argmax() 
            df_candidate = df_eq_not_None.iloc[idx_err_min]
            idx_err_min = pd.to_numeric(little_df_eq["r2_val"]).argmax()
            df_little = little_df_eq.iloc[idx_err_min]

            total_r2_candidates += [df_candidate["r2_test"]]
            total_r2_vanillas += [vanilla["r2_test"]]
            total_r2_little += [df_little["r2_test"]]
            #print("PointWise Candidate:", df_candidate["pointwise"])
            total_pointwise_candidates += [df_candidate["pointwise"]]
            #print("PointWise Vanilla:", vanilla["pointwise"])
            total_pointwise_vanillas += [vanilla["pointwise"]]
            total_poinwise_little += [df_little["pointwise"]]
            seeds += [seed]
            eq_name += [df_eq["eq_name"].iloc[0]]
    
    df_candidate = pd.DataFrame({"r2": total_r2_candidates, "pointwise": total_pointwise_candidates, "seed": seeds, "eq_name": eq_name})
    df_vanilla = pd.DataFrame({"r2": total_r2_vanillas, "pointwise": total_pointwise_vanillas,  "seed": seeds, "eq_name": eq_name})
    df_little = pd.DataFrame({"r2": total_r2_little, "pointwise": total_poinwise_little,  "seed": seeds, "eq_name": eq_name})
    df_candidate["type"] = "candidate"
    df_vanilla["type"] = "vanilla"
    df_little["type"] = "little"
    df = pd.concat([df_candidate, df_vanilla, df_little])
    df["r2"] = pd.to_numeric(df["r2"])
    df["pointwise"] = pd.to_numeric(df["pointwise"])
    df["pointwise"] = df["pointwise"]*100
    df["r2"] = df["r2"]
    df["r2"] = df["r2"].round(2)
    # Clip to 0 the r2
    df["r2"] = df["r2"].clip(lower=0)

    df["pointwise"] = df["pointwise"].round(2)
    # Reset the index
    df = df.reset_index(drop=True)
    df["is_correct"] = df["pointwise"] > 99.9999999 #& (~df["eq_name"].isin(["strogatz_predprey1","strogatz_vdp2"]))
    group = df.groupby(["seed", "type"]).mean().reset_index()
    # Compute the mean and the std
    group_mean = group.groupby(["type"]).mean().reset_index()
    group_std = group.groupby(["type"]).std().reset_index()

    group_mean["type"] = pd.Categorical(group_mean["type"], categories=["candidate", "vanilla", "little"])
    group_mean = group_mean.sort_values("type")
    group_std["type"] = pd.Categorical(group_std["type"], categories=["candidate", "vanilla", "little"])
    group_std = group_std.sort_values("type")

    group_mean["r2"] = group_mean["r2"].round(2)
    group_std["r2"] = group_std["r2"].round(2)
    group_mean["is_correct"] = group_mean["is_correct"].round(2)
    group_std["is_correct"] = group_std["is_correct"].round(2)

    group_mean = group_mean.rename(columns={ "is_correct": "Correct","r2": "R2"})
    group_std = group_std.rename(columns={"is_correct": "Correct", "r2": "R2", })
    group_mean = group_mean[["type", "Correct", "R2", ]]
    group_std = group_std[["type", "Correct", "R2", ]]
    group_mean = group_mean.to_latex(index=False)
    group_std = group_std.to_latex(index=False)
    print(group_mean)
    print(group_std)


def main():
    #table_third_exp()
    plot_controllability_main_body()
    plot_main_accuracy_presentation(metric="ood_pointwise_acc", is_vanilla_vs_nesy=True)
    plot_main_accuracy_presentation(metric="ood_r2", is_vanilla_vs_nesy=True)

    plot_main_accuracy_points_noise("points",metric="ood_pointwise_acc", is_vanilla_vs_nesy=False)
    plot_main_accuracy_points_noise("noise",metric="ood_pointwise_acc", is_vanilla_vs_nesy=False)
    plot_main_accuracy_points_noise("points",metric="ood_r2", is_vanilla_vs_nesy=False)
    plot_main_accuracy_points_noise("noise",metric="ood_r2", is_vanilla_vs_nesy=False)

    #plot_main_accuracy_points_noise("points",metric="ood_pointwise_acc", is_vanilla_vs_nesy=True)
    #plot_main_accuracy_points_noise("noise",metric="ood_pointwise_acc", is_vanilla_vs_nesy=True)
    #plot_main_accuracy_points_noise("points",metric="ood_r2", is_vanilla_vs_nesy=True)
    #plot_main_accuracy_points_noise("noise",metric="ood_r2", is_vanilla_vs_nesy=True)
    
    

if __name__ == "__main__":
    main()