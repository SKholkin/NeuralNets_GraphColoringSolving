# NeuralNets_GraphColoringSolving
Solving a dicision version of Graph Coloring Problem (GCP) by DeepLearning approach

## Creating venv
```
pip install -r requirements.txt
```
Also you can install cuda version of torch and run training on GPU (though it doesn't give much speed up)
## Data
To create synthetic data (hard to color graph instances) for training
```
python dataset_generator.py --samples number_of_samples --nmin min_number_of_vertices --nmax max_number_of_vertcies --root path_to_dataset --test is_test
```
Also you can convert dataset from https://machine-reasoning-ufrgs.github.io/GNN-GCP/ through running:
```
python convert_gnn_gcp_dataset.py path_to_downloaded_dataset
```
## Model
Currently available Reccurent Graph Neural Network model similar to https://arxiv.org/abs/1903.04598 with various attention mechanisms and lack of layer normalization.
### Training and evaluating
```
python --mode train_or_test --config path_to_your_config --data path_to_dataset --print_freq print_freq --log_dir dir_to_save_checkpoints_and_log --save_freq save_freq --test_freq test_freq
```
### Attention aggegators
You can change version of attention aggregators through setting such option in config:
```
{
  "attention": true,
  "attention_version": "pairwise_0"
}
```
There are available "pairwise_0", "pairwise_1" and "pairwise_2" aggergators (named in ascending of computational complexity)
Basic graph attention aggregator related paper (https://arxiv.org/abs/1710.10903)
## Current results
|Config|Data generation model|Data vertices number|Data chrom number|Acc|
  | :--- | :---: | :---: | :---: | :---: |
  |[Basic](./configs/basic.json)| [This](./dataset_generator.py) | 35-45 | 3-8 | 87.3 |
  |[Pairwise_0](./configs/basic_pairwise_0.json)| [This](./dataset_generator.py) | 35-45 | 3-8 | 87.3 |
 
Experiments on GNN_GCP (https://machine-reasoning-ufrgs.github.io/GNN-GCP/) haven't gone well (more likely there is bug somewhere)
### Work on project has been stopped
