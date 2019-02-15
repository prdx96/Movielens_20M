import os
import json
import numpy as np
import pandas as pd

def export_result(modelname, rmse, top_k, recall, f1, precision, dataset, executed_at='test', resultfile='./log/result.csv'):
  result = ','.join([modelname, str(rmse), str(top_k), str(recall), str(f1), str(precision), dataset, executed_at])
  with open(resultfile, 'a') as f:
    f.write(result)
    f.write('\n')

def export_learning_process(train_rmse, validation_rmse, test_rmse, execute_time='test'):
  train = np.array(train_rmse).reshape(-1,1)
  validation = np.array(validation_rmse).reshape(-1,1)
  test = np.array(test_rmse).reshape(-1,1)

  learning_process = pd.DataFrame(np.concatenate([train, validation, test], axis=1), columns=['train', 'validation', 'test'])
  filename = execute_time + '_learning_process.csv'
  if not os.path.exists('./log/learning_process'):
    os.mkdir('./log/learning_process')
  filepath = os.path.join('./log/learning_process', filename)
  learning_process.to_csv(filepath, index=None)

def load_json(filename):
  with open(filename) as f:
    json_dict = json.load(f)
  return json_dict
