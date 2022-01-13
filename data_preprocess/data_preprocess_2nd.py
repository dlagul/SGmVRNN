import numpy as np
import argparse
import pandas as pd
import os, sys
import math
import string
import torch
import time

def data_preprocess(raw_data_file,label_file,data_path,
                    sample_id = 1,win_size=36,l=10,T=20,fr=(-1,1)):
    if not os.path.exists(raw_data_file):
        raise ValueError('Unknown input data file: {}'.format(raw_data_file))
    if not os.path.exists(label_file):
        raise ValueError('Unknown input label file: {}'.format(label_file))
    # get raw data and the corresponding labels
    scaled_data = np.array(pd.read_csv(raw_data_file, header = 0), dtype=np.float64)
    raw_ts_label = np.array(pd.read_csv(label_file, header = None), dtype=np.int64)
    raw_ts = raw_ts_label[0,:]
    raw_ts = np.array([raw_ts.tolist()])
    raw_label = raw_ts_label[1,:]
    raw_label = np.array([raw_label.tolist()])
    
    rectangle_samples = []
    rectangle_labels = []
    rectangle_tss = []

    for j in range(l):
        rectangle_sample = []
        rectangle_label = []
        rectangle_ts = []
        for i in range(0, scaled_data.shape[1]-win_size, l):
            if i+j <= scaled_data.shape[1]-win_size:
                scaled_data_tmp = scaled_data[:,i+j:i+j+win_size]
                rectangle_sample.append(scaled_data_tmp.tolist())
                raw_label_tmp = raw_label[:,i+j:i+j+win_size]
                rectangle_label.append(raw_label_tmp.tolist())
                raw_ts_tmp = raw_ts[:,i+j:i+j+win_size]
                rectangle_ts.append(raw_ts_tmp.tolist())

        rectangle_samples.append(np.array(rectangle_sample))
        rectangle_labels.append(np.array(rectangle_label))
        rectangle_tss.append(np.array(rectangle_ts))
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for i in range(len(rectangle_samples)):
        for data_id in range(T, len(rectangle_samples[i])):
            kpi_data = rectangle_samples[i][data_id-T:data_id] 
            kpi_label = rectangle_labels[i][data_id-T:data_id]
            kpi_ts = rectangle_tss[i][data_id-T:data_id]
            kpi_data = torch.tensor(kpi_data).unsqueeze(1)
            data = {'ts':kpi_ts,
                'label':kpi_label,
                'value':kpi_data}
            cur_timestamp = kpi_ts[-1][-1][-1]
            cur_time_stamp = int(time.mktime(time.strptime(str(cur_timestamp), '%Y%m%d%H%M%S')))
            
            path_temp = os.path.join(data_path, str(sample_id))
            torch.save(data, path_temp + '.seq')
            sample_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_file', type=str, default='')
    parser.add_argument('--label_file', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--sample_id', type=int, default=1)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--win_size', type=int, default=1)
    parser.add_argument('--l', type=int, default=1)

    args = parser.parse_args()
    
    data_preprocess(args.raw_data_file,
                    args.label_file,
                    args.data_path,
                    sample_id = args.sample_id,
                    win_size  = args.win_size,
                    l         = args.l,
                    T         = args.T)

if __name__ == '__main__':
    main()
