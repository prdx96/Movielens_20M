import json
import os
import sys
from time import time
import argparse

from scipy import sparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file

def parse_args():
    parser = argparse.ArgumentParser(description="Convert csv file to libfm formatted file")
    parser.add_argument('--path', nargs='?', default='../data/org/ratings.csv',
                        help='Input data path.')
    parser.add_argument('--target_dir', nargs='?', default='../data/preprocessed', help='Output data path')
    return parser.parse_args()


class CreateLibfmFile():
  '''
  create .libfm formatted file

  self.data: original csv file(ratings.csv)
  self.target_dir: the directory libfm files should by exported to

  Input: One csv file
  Output: Three libfm files(train.libfm, validation.libfm, test.libfm)
  '''
  def __init__(self, org_filename, target_dir):
    self.data = pd.read_csv(org_filename)
    self.data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    self.data = self.data.iloc[:, :-1]
    self.target_dir = target_dir

    self.train_data = self.data.iloc[:14000000,:]
    self.validation_data = self.data.iloc[14000000:18000000,:]
    self.test_data = self.data.iloc[18000000:,:]

    self.user_len = None
    self.user_idx_dict = None
    self.idx_user_dict = None
    self.item_len = None
    self.item_idx_dict = None
    self.idx_item_dict = None
    self.create_dict()

  def create_dict(self):
    '''
    create user-item dictionary and save them in the data directory
    '''
    user_list = self.data.user_id.unique()
    self.user_len = len(user_list)
    # TODO: fix the way of creating dict(both k and v should be stored as int)
    self.user_idx_dict = {str(user_list[j]): str(j) for j in range(0, self.user_len)}
    self.idx_user_dict = {str(j): str(user_list[j]) for j in range(0, self.user_len)}

    item_list = self.data.item_id.unique()
    self.item_len = len(item_list)
    self.item_idx_dict = {str(item_list[i]):str(i + self.user_len) for i in range(0, self.item_len)}
    self.idx_item_dict = {str(i + self.user_len):str(item_list[i]) for i in range(0, self.item_len)}

    if not os.path.exists(self.target_dir):
      os.mkdir(self.target_dir)

  def dict2json(self, dict, filename):
    with open(filename, 'w') as f:
        json.dump(dict, f)

  def export_dict(self):
    if not os.path.exists(os.path.join(self.target_dir, 'json/')):
      os.mkdir(os.path.join(self.target_dir, 'json/'))
    filename_user_idx = os.path.join(self.target_dir, 'json/user_idx_dict.json')
    self.dict2json(self.user_idx_dict, filename_user_idx)

    filename_idx_user = os.path.join(self.target_dir, 'json/idx_user_dict.json')
    self.dict2json(self.idx_user_dict, filename_idx_user)

    filename_item_idx = os.path.join(self.target_dir, 'json/item_idx_dict.json')
    self.dict2json(self.item_idx_dict, filename_item_idx)
    filename_idx_item = os.path.join(self.target_dir, 'json/idx_item_dict.json')
    self.dict2json(self.idx_item_dict, filename_idx_item)

  def create_sp_matrix(self, df):
    '''
    create a sparse matrix from a DataFrame

    Input: pd.DataFrame
    Return: csr_matrix, ndarray
    '''
    sp_matrix = sparse.lil_matrix((len(df), self.user_len + self.item_len), dtype=np.int8)
    label = []

    for i in range(len(df)):
        sp_matrix[int(i), int(self.user_idx_dict[str(df.iloc[i,0])])] = 1
        sp_matrix[int(i), int(self.item_idx_dict[str(df.iloc[i,1])])] = 1
        label.append(df.iloc[i,2])

    sp_matrix.tocsr()
    #label = np.ones(len(df), dtype=np.int8)
    label = np.array(label, dtype=np.float32)
    return sp_matrix, label

  def convert(self):
    # create sparse.csr_matrix and ndarray
    X_train, y_train = self.create_sp_matrix(self.train_data)
    filename_train = os.path.join(self.target_dir, 'train.libfm')
    dump_svmlight_file(X_train, y_train, filename_train)

    X_validation, y_validation = self.create_sp_matrix(self.validation_data)
    filename_validation = os.path.join(self.target_dir, 'validation.libfm')
    dump_svmlight_file(X_validation, y_validation, filename_validation)

    X_test, y_test = self.create_sp_matrix(self.test_data)
    filename_test = os.path.join(self.target_dir, 'test.libfm')
    dump_svmlight_file(X_test, y_test, filename_test)


if __name__=='__main__':
  start_time = time()
  print('Start data preprocessing...')
  args = parse_args()

  clibfm = CreateLibfmFile(args.path, args.target_dir)
  clibfm.create_dict()

  print('Export dict to json files...')
  clibfm.export_dict()

  print('Creating libfm files...')
  clibfm.convert()

  end_time = time()
  print('Finished in', str(end_time - start_time), ' sec.')
  print('Done!')
