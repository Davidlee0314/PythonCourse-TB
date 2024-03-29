'''
Usage Example : 
    python3.7 test_threshold.py -p ./infer_val/InferVal_Net1D_2_Focal_a075_g3_t015_b2048_ep8.pkl 
'''
import argparse

import pandas as pd
import pickle as pkl

from confusion import cm_f1_score

def test_threshold(infer_val_path = './test.pkl'):
    with open(infer_val_path, 'rb') as file:
        infer_val_df = pkl.load(file)

    print(infer_val_df.head(), '\n')

    threshold_list = [ 0.05 + 0.025 * x for x in range(0, 15)]
    threshold_f1_dict = {}
    for threshold in threshold_list:
        # print('threshold_{}'.format(str(threshold)[2:6]))
        infer_val_df['threshold_{}'.format(str(threshold)[2:6])] = infer_val_df['pred_softmax'] > threshold
        infer_val_df['threshold_{}'.format(str(threshold)[2:6])] = infer_val_df['threshold_{}'.format(str(threshold)[2:6])].astype(int)
        f1 = cm_f1_score(infer_val_df['label'], infer_val_df['threshold_{}'.format(str(threshold)[2:6])], verbose=False)
        threshold_f1_dict['threshold_{}'.format(str(threshold)[2:6])] = f1
    for ind, val in threshold_f1_dict.items():
        print('{} : f1 = {}'.format(ind, val))
    max_f1 = max(threshold_f1_dict.values())
    max_key = list(threshold_f1_dict.keys())[list(threshold_f1_dict.values()).index(max_f1)]
    print('\nMax f1 score = {}   (when {})\n'.format(max_f1, max_key))

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_val_path", "-p", type=str, default='./test.pkl', help="pickle file path to load predicted output")

    opt = parser.parse_args()
    print('\n', opt)
    return opt


if __name__ == "__main__":
    opt = args_parse()
    test_threshold(opt.infer_val_path)

    # m = max(a.values())
    # k = list(a.keys())[list(a.values()).index(m)]