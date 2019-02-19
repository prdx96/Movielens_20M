import argparse
from time import time
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
import trainer.LoadData as DATA
from trainer.NeuralFM import NeuralFM


#################### Arguments ####################
def parse_args():
  parser = argparse.ArgumentParser(description="Run Neural FM.")
  parser.add_argument(
    '--job-dir',
    type=str,
    help='GCS or local dir to write checkpoints and export model',
    default='gs://dev-michishita-recommend/tmp'
    )
  parser.add_argument('--path', nargs='?', default='data/',
                      help='Input data path.')
  parser.add_argument('--epoch', type=int, default=200,
                      help='Number of epochs.')
  parser.add_argument('--pretrain', type=int, default=0,
                      help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
  parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size.')
  parser.add_argument('--hidden_factor', type=int, default=64,
                      help='Number of hidden factors.')
  parser.add_argument('--layers', nargs='?', default='[64]',
                      help="Size of each layer.")
  parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.5]',
                      help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
  parser.add_argument('--lamda', type=float, default=0,
                      help='Regularizer for bilinear part.')
  parser.add_argument('--lr', type=float, default=0.05,
                      help='Learning rate.')
  parser.add_argument('--loss_type', nargs='?', default='square_loss',
                      help='Specify a loss type (square_loss or log_loss).')
  parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                      help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
  parser.add_argument('--verbose', type=int, default=1,
                      help='Show the results per X epochs (0, 1 ... any positive integer)')
  parser.add_argument('--batch_norm', type=int, default=1,
                  help='Whether to perform batch normaization (0 or 1)')
  parser.add_argument('--activation', nargs='?', default='relu',
                  help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
  parser.add_argument('--early_stop', type=int, default=1,
                  help='Whether to perform early stop (0 or 1)')
  return parser.parse_args()


if __name__=='__main__':
  args = parse_args()

  hparams = hparam.HParams(**args.__dict__)
  data = DATA.LoadData(  hparams.path,  hparams.loss_type)
  if hparams.verbose > 0:
      print("Neural FM: hidden_factor=%d, dropout_keep=%s, layers=%s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
            %(hparams.hidden_factor, hparams.keep_prob, hparams.layers, hparams.loss_type, hparams.pretrain, hparams.epoch, hparams.batch_size, hparams.lr, hparams.lamda, hparams.optimizer, hparams.batch_norm, hparams.activation, hparams.early_stop))
  activation_function = tf.nn.relu
  if hparams.activation == 'sigmoid':
      activation_function = tf.sigmoid
  elif hparams.activation == 'tanh':
      activation_function == tf.tanh
  elif hparams.activation == 'identity':
      activation_function = tf.identity

  # Training
  t1 = time()
  model = NeuralFM(data.features_M, hparams.hidden_factor, eval(hparams.layers), hparams.loss_type, hparams.pretrain, hparams.epoch, hparams.batch_size, hparams.lr, hparams.lamda, eval(hparams.keep_prob), hparams.optimizer, hparams.batch_norm, activation_function, hparams.verbose, hparams.early_stop)
  model.train(data.Train_data, data.Validation_data, data.Test_data)

  # Find the best validation result across iterations
  best_valid_score = 0
  if hparams.loss_type == 'square_loss':
      best_valid_score = min(model.valid_rmse)
  elif hparams.loss_type == 'log_loss':
      best_valid_score = max(model.valid_rmse)
  best_epoch = model.valid_rmse.index(best_valid_score)
  print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
         %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch], time()-t1))
