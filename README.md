# PSPS: Preconditioned Stochastic Polyak Step-size method for badly scaled data

The family of Stochastic Gradient Methods with Polyak Step-size offers an update rule that alleviates the need of fine-tuning the learning rate of an optimizer. Recent work has been proposed to introduce a slack variable, which makes these methods applicable outside of the interpolation regime. In this paper, we combine preconditioning and slack in an updated optimization algorithm to show its performance on badly scaled and/or ill-conditioned datasets. We use Hutchinson’s method to obtain an estimate of a Hessian which is used as the preconditioner.


## How to run experiments

### 1. Create and environment and install dependencies
Use the package manager [conda](https://docs.conda.io/en/latest/) to create an environment and install all dependencies from ```environment.yaml``` provided in this package. 
```console
conda env create -n ENVNAME --file environment.yaml
```

### 2. Create ```.env```
To manage environment variables with ```dotenv``` create ```.env``` file in the root directory of the package and specify absolute path to the following folders:
```bash
RESULTS_DIR=PATH_TO_EXPERIMENT_RESULTS_DIRECTORY
PLOTS_DIR=PATH_TO_PLOTS_DIRECTORY
DATASETS_DIR=PATH_TO_DATASETS_DIRECTORY
NN_TENSORBOARD_DIR=PATH_TO_TENSORBOARD_RUNS_FOR_NN_EXPERIMENTS
```

### 3. Create directories specified in ```.env``` and download datasets from [LibSVM Datasets Repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
This example creates a directory for datasets and downloads ```colon-cancer``` dataset from LibSVM Datasets Reposotory. Since this dataset is compressed we use ```bzip2``` to uncompress it. Note that ```MNIST``` dataset is fetched from ```PyTorch``` datasets package so you do not need to download it manually.
```console
user@user:~$ mkdir PATH_TO_DATASETS_DIRECTORY
user@user:~$ cd PATH_TO_DATASETS_DIRECTORY
user@user:~$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/colon-cancer.bz2
user@user:~$ ls -la
drwxr-xr-x 2 root user      4 Nov 21 10:15 .
drwxr-xr-x 4 root user     16 Nov 21 10:13 ..
-rw-r--r-- 1 root user 609480 Sep  3  2007 colon-cancer.bz2
user@user:~$ bzip2 -d colon-cancer.bz2
user@user:~$ ls -la 
drwxr-xr-x 2 root user      4 Nov 21 10:15 .
drwxr-xr-x 4 root user     16 Nov 21 10:13 ..
-rw-r--r-- 1 root user 1727461 Sep  3  2007 colon-cancer
```

### 4. Run experiments with Logistic Regression and NLLSQ
To see help message for ```run.py``` arguments execute:
```console
python run.py --help
```
The following is an example run of ```run.py```:

```console
python run.py --dataset=mushrooms --batch_size=64 --epochs=100 --loss=logreg --optimizer=sps --preconditioner=hutch --slack=L1 --seed=1 --save --no-tb
```
The above example will run an experiment on ```mushrooms``` dataset with batch size of 64 samples, for 100 epochs, with Logistic Regression loss function, optimized by preconditioned ```SPS``` with ```L1``` slack method. All ```PyTorch``` random values will be generated with ```torch.random.manual_seed(seed)```. Experiment results will be saved to ```RESULTS_DIR``` directory specified in ```.env``` but no ```TensorBoard``` run will be created. 


### 5. Run experiments with Neural Network

To see help message for ```run_nn.py``` arguments execute:
```console
python run_nn.py --help
```
The following is an example run of ```run_nn.py```:

```console
python run_nn.py --dataset=MNIST --model=smlenet --batch_size=512 --epochs=100 --loss=nll_loss --optimizer=sps --preconditioner=hutch --slack=L1 --seed=1 --save --no-tb
```
The above example will run an experiment with ```SMLENET``` (Small LeNet) model trained on ```MNIST``` dataset with batch size of 512 samples, for 100 epochs, with NLLSQ loss function, optimized by preconditioned ```SPS``` with ```L1``` slack method. All ```PyTorch``` random values will be generated with ```torch.random.manual_seed(seed)```. Experiment results will be saved to ```RESULTS_DIR``` directory specified in ```.env``` but no ```TensorBoard``` run will be created. 


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.