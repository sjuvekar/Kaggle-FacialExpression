import numpy
from sklearn import preprocessing
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from theano import tensor as T
import theano

class ConvolutionModel(object):

    def __init__(self):
      
        train="""
          !obj:pylearn2.train.Train {
            dataset: &train !obj:pylearn2.datasets.csv_dataset.CSVDataset {
              path: 'csv/train.csv'
            },
            model: !obj:pylearn2.models.mlp.MLP {
              input_space: !obj:pylearn2.space.Conv2DSpace {
                       shape: [48, 48],
                       num_channels: 1
              },
              layers: [
                       !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                           layer_name: 'h2',
                           output_channels: 64,
                           kernel_shape: [8, 8],
                           pool_shape: [4, 4],
                           pool_stride: [2, 2],
                           irange: .05,
                           max_kernel_norm: 0.9365
                       },
                       !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                           layer_name: 'h3',
                           output_channels: 64,
                           kernel_shape: [5, 5],
                           pool_shape: [2, 2],
                           pool_stride: [1, 1],
                           irange: .05,
                           max_kernel_norm: 0.9365
                       },
                       !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                           layer_name: 'h4',
                           output_channels: 64,
                           kernel_shape: [5, 5],
                           pool_shape: [4, 4],
                           pool_stride: [2, 2],
                           irange: .05,
                           max_kernel_norm: 0.9365
                       },
                       !obj:pylearn2.models.mlp.Sigmoid {
                           layer_name: 'h0',
                           dim: 5000,
                           irange: .05
                       },
                       !obj:pylearn2.models.mlp.Sigmoid {
                           layer_name: 'h1',
                           dim: 5000,
                           irange: .05
                       },
                       !obj:pylearn2.models.mlp.Softmax {
                           layer_name: 'y',
                           n_classes: 7,
                           istdev: 0.05,
                           max_col_norm: 0.9365
                       }
                      ]
            },
            algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
              batch_size: 100,
              learning_rate: 0.001,
              learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
                  init_momentum: 0.5,
              },
              monitoring_dataset:
                  {
                      'train' : *train,
                      'valid' : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                              path: 'csv/valid.csv'
                      },
                      'test'  : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                              path: 'csv/test.csv'
                      }
                  },
              termination_criterion: !obj:pylearn2.termination_criteria.And {
                  criteria: [
                      !obj:pylearn2.termination_criteria.EpochCounter {
                          max_epochs: 500
                      }
                  ]
              }
            },
            extensions: [
              !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                   channel_name: 'valid_y_misclass',
                   save_path: "cnn_best.pkl"
              },
            ]
          }
          """
        self.classifier = yaml_parse.load(train)
	self.model_path = "cnn_best.pkl"


    def predict(self):
        model = serial.load(self.model_path)
        X = model.get_input_space().make_theano_batch()
        Y = model.fprop( X )
        Y = T.argmax( Y, axis = 1 )
        f = theano.function( [X], Y )
	x_len = len(self.X_test)
        return f(numpy.array(self.X_test).reshape(x_len, 60, 60, 1).astype(numpy.float32))


if __name__ == "__main__":
    c = ConvolutionModel()
    c.classifier.main_loop() 
