import numpy as np
import os
import tensorflow as tf

class LoadData(object):
  '''given the path of data, return the data format for DeepFM
  :param path
  return:
  Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
  Test_data: same as Train_data
  Validation_data: same as Train_data
  '''

  # Three files are needed in the path
  def __init__(self, path, loss_type):
    self.path = path
    self.trainfile = self.path +"train.libfm"
    self.testfile = self.path + "test.libfm"
    self.validationfile = self.path + "validation.libfm"
    self.features_M = self.map_features( )
    self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )

  def map_features(self): # map the feature entries in all files, kept in self.features dictionary
    self.features = {}
    self.read_features(self.trainfile)
    self.read_features(self.testfile)
    self.read_features(self.validationfile)
    #print("features_M:", len(self.features))
    return  len(self.features)

  def read_features(self, file): # read a feature file
    f = tf.gfile.Open(file, 'rb')
    line = f.readline()
    i = len(self.features)
    while line:
      items = line.decode("utf-8").strip().split(' ')
      if i == 0:
        print(items[0])
        print(items[1])
        print(items[2])
      for item in items[1:]:
        if item not in self.features:
          self.features[ item ] = i
          i = i + 1
      line = f.readline()
    f.close()

  def construct_data(self, loss_type):
    X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
    if loss_type == 'log_loss':
      Train_data = self.construct_dataset(X_, Y_for_logloss)
    else:
      Train_data = self.construct_dataset(X_, Y_)
    print("# of training:" , len(Y_))

    X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
    if loss_type == 'log_loss':
      Validation_data = self.construct_dataset(X_, Y_for_logloss)
    else:
      Validation_data = self.construct_dataset(X_, Y_)
    print("# of validation:", len(Y_))

    X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
    if loss_type == 'log_loss':
      Test_data = self.construct_dataset(X_, Y_for_logloss)
    else:
      Test_data = self.construct_dataset(X_, Y_)
    print("# of test:", len(Y_))

    return Train_data,  Validation_data,  Test_data

  def read_data(self, file):
    # read a data file. For a row, the first column goes into Y_;
    # the other columns become a row in X_ and entries are maped to indexs in self.features
    f = tf.gfile.Open(file, 'rb')
    X_ = []
    Y_ = []
    Y_for_logloss = []
    line = f.readline()
    while line:
      items = line.decode("utf-8").strip().split(' ')
      Y_.append( 1.0*float(items[0]) )

      if float(items[0]) > 0:# > 0 as 1; others as 0
        v = 1.0
      else:
        v = 0.0
      Y_for_logloss.append( v )

      X_.append( [ self.features[item] for item in items[1:]] )
      line = f.readline()
    f.close()
    return X_, Y_, Y_for_logloss

  def construct_dataset(self, X_, Y_):
    Data_Dic = {}
    X_lens = [ len(line) for line in X_]
    indexs = np.argsort(X_lens)
    Data_Dic['Y'] = [ Y_[i] for i in indexs]
    Data_Dic['X'] = [ X_[i] for i in indexs]
    return Data_Dic
