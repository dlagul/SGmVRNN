# SGmVRNN
SGmVRNN is a switching Gaussian mixture variational recurrent neural network based anomaly detection for diverse CDN websites.

## Getting Started
### Install dependencies (with python 3.5 or 3.6)
#### Python virtual environment is suggested

    virtualenv -p <Your python path> venv
    source venv/bin/activate
    pip install -r requirements.txt
    
### Data Preprocessing
The data preprocessing is split into two steps: including data normalisation and data slicing by sliding window.
##### Note that for 'data_preprocess_1st.py' and 'data_preprocess_2nd.py', please refer to the folder 'data_preprocess'.

    cd data_preprocess
#### 1st step:
##### For SMD
    python data_preprocess_1st.py --train_path SMD/train --test_path SMD/test --label_path SMD/test_label --output_path SMD_concated --dataset SMD
#### 2nd step:
##### For SMD training set
    python data_preprocess_2nd.py --raw_data_file SMD_concated/machine_kpi_ts_data_train.csv --label_file SMD_concated/machine_kpi_ts_label_train.csv --data_path data_processed/smd-train &
    
##### For SMD testing set
    python data_preprocess_2nd.py --raw_data_file SMD_concated/machine-1-1_data_test.csv --label_file SMD_concated/machine-1-1_label_test.csv --data_path data_processed/machine-1-1 &
    
##### Note that:
The processed training set for SMD can be found in data_preprocess/data_processed/smd-train;
The processed training set for all of the machines can be found in data_preprocess/data_processed, e.g., data_preprocess/data_processed/machine-1-1, etc.

### Running SGmVRNN

    cd SGmVRNN

### Training
##### For SMD
    python trainer.py --dataset_path ../data_preprocess/data_processed/smd-train --gpu_id 1 --log_path log_trainer/smd --checkpoints_path model/smd --n 38
 
### Testing
##### For SMD 
###### e.g., for machine-1-1
    nohup python tester.py --dataset_path ../data_preprocess/data_processed/machine-1-1 --gpu_id 1 --log_path log_tester/machine-1-1 --checkpoints_path model/smd --n 38 2>&1 &
 
### Evaluation via POT
##### For SMD
###### e.g., for machine-1-1
    nohup python evaluate_pot.py --llh_path log_tester/machine-1-1 --log_path log_evaluator_pot/machine-1-1 --n 38 2>&1 &

##### The Final Results via POT
The final results including F1-score, Precision, Recall, etc. are in "SGmVRNN/log_evaluator_pot". <br>

## Datasets
### Basic Statistics
Statistics | KPIs of CDN  | KPIs of SMD 
--- | --- | --- 
Dimensions | 31 * 36 | 28 * 38
Granularity (sec) | 60 | 60 
Training set size | 1,227,249 | 708,405 
Testing set size | 1,227,250 | 708,420 
Anomaly Ratio (%) | 3.68 | 4.16 

### CDN Dataset Information

#### CDN Data Format
There are 3 comma separated CSV files of each website,including training, testing set as well as the corresponding ground-truth file of testing set. <br>
The KPIs data file has the following format: <br>
* The columns are the KPIs values, and each column corresponds to a KPI <br>

 KPI_1 | KPI_2 | ... | KPI_n
 --- | --- | --- | ---
 0.5 | 0.6 | ... | 0.7
 0.3 | 0.4 | ... | 0.5
 ... | ... | ... | ... 
 0.9 | 0.8 | ... | 0.7

The ground-truth file has the following format: <br>
* The column is the label, 0 for normal and 1 for abnormal <br>

 |label|  
 |---| 
 | 0 |
 | 1 |
 | ... |
 | 0 |

### Public Dataset 

Please refer to https://github.com/NetManAIOps/OmniAnomaly for public SMD dataset.
