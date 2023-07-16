import setuptools 

setuptools.setup(
      version='0.1.0',
      description='Code for reproducing the results of Controllable Neural Symbolic Regression that scales',
      name="ControllableNesymres",
      packages=setuptools.find_packages('.'),
      package_dir={'': '.'},
      install_requires=[
          'numpy<1.20.0',
          'sympy==1.11.1',
          'pytorch_lightning<2.0.0',
          'joblib',
          'pandas<1.5','timeout_decorator','click', 
          'tqdm','numexpr','jsons', 
          "h5py","scipy","dataclass_dict_convert", 
          "hydra-core==1.2.0", 
          "ordered_set", "streamlit", "seaborn","scikit-learn","tensorboard",
          "pmlb","huggingface_hub"
      ]
     )
