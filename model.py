from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics

tf.logging.set_verbosity(tf.logging.INFO)

BUCKET = 'price_pred'
PROJECT = 'Dynamic Pricing - ML'
REGION = 'US-CENTRAL1'

import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION


#BUCKET = None  # set from task.py
#PATTERN = 'of' # gets all files
#TRAIN_STEPS = 10000

CSV_COLUMNS = 'NET_COST,Style_Name,Quantity,Demand,Original_Retail_Price,Selling_Price,Margin,off_Orig_Retail,Total_OTS'.split(',')
LABEL_COLUMN = 'NET_COST'
DEFAULTS = [[0.0], ['null'], [0], [0], [0], [0], [0], [0], [0]]

def read_dataset(prefix, pattern, batch_size=512):
    filename = 'gs://{}/price_pred/input/{}*{}*'.format(BUCKET, prefix, pattern)
    if prefix == 'Training':
        mode = tf.contrib.learn.ModeKeys.TRAIN
    else:
        mode = tf.contrib.learn.ModeKeys.EVAL
    
  # the actual input function passed to TensorFlow
    def _input_fn():
    # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)
 
    # read CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        #features.pop(KEY_COLUMN)
        label = features.pop(LABEL_COLUMN)
        return features, label
  
    return _input_fn

def get_wide_deep():
    # define column types
    
    StyleName,quantity, demand, org_ret_price,sell_price, margin, off_orig_retail, total_ots = \
    [ \
    tflayers.sparse_column_with_hash_bucket('Style_Name', hash_bucket_size = 1000),
    tflayers.real_valued_column('Quantity'),
    tflayers.real_valued_column('Demand'),
    tflayers.real_valued_column('Original_Retail_Price'),
    tflayers.real_valued_column('Selling_Price'),
    tflayers.real_valued_column('Margin'),
    tflayers.real_valued_column('off_Orig_Retail'),
    tflayers.real_valued_column('Total_OTS'),
    ]
    # which columns are wide (sparse, linear relationship to output) and which are deep (complex relationship to output?)  
    wide = [StyleName,quantity, demand]
    deep = [\
               org_ret_price,
               sell_price,
               margin,
               off_orig_retail,
               total_ots,
               tflayers.embedding_column(StyleName, 3)
               ]
    return wide, deep

def serving_input_fn():
    feature_placeholders = {
      'Style_Name': tf.placeholder(tf.string, [None]),
      'Quantity': tf.placeholder(tf.int32, [None]),
      'Demand': tf.placeholder(tf.int32, [None]),
      'Original_Retail_Price': tf.placeholder(tf.float32, [None]),
      'Selling_Price': tf.placeholder(tf.float32, [None]),
      'Margin': tf.placeholder(tf.int32, [None]),
      'off_Orig_Retail': tf.placeholder(tf.int32, [None]),
      'Total_OTS': tf.placeholder(tf.float32, [None])
    }
    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
    }
    return tflearn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders)
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
def experiment_fn(output_dir):
    wide, deep = get_wide_deep()
    return tflearn.Experiment(
        tflearn.DNNLinearCombinedRegressor(model_dir=output_dir,
                                           linear_feature_columns=wide,
                                           dnn_feature_columns=deep,
                                           dnn_hidden_units=[64, 32]),
        train_input_fn=read_dataset('Training'),
        eval_input_fn=read_dataset('Evaluation'),
        eval_metrics={
            'rmse': tflearn.MetricSpec(
                metric_fn=metrics.streaming_root_mean_squared_error
            )
        },
        export_strategies=[saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        train_steps=TRAIN_STEPS
)