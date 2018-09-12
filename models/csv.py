import tensorflow as tf
import itertools
import random
import numpy as np
import logging
import glob
import pandas as pd
import shutil
import os


def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def _train_output_format(x_variables, x_exogenous, x_times, y_variables, y_exogenous, y_times):
    return {'inputs': (x_variables, x_exogenous, x_times), 'outputs': (y_exogenous, y_times)}, y_variables


def _test_output_format(x_variables, x_exogenous, x_times, y_exogenous, y_times):
    return {'inputs': (x_variables, x_exogenous, x_times), 'outputs': (y_exogenous, y_times)}


def submit_input_fn(train, test, input_window_size, output_window_size):
    import datetime
    train = pd.read_csv(train, parse_dates=False)
    test = pd.read_csv(test, parse_dates=False)

    def _date(v):
        return datetime.datetime.strptime(v, "%Y-%m-%d").date()

    start = _date('2010-01-01')
    train_data = {}
    predict_data = {}
    for name, group in train.groupby(['store', 'item']):
        group = group.copy(deep=False)
        group.index = range(len(group))
        group['time'] = group['date'].apply(lambda x: (_date(x) - start).days)
        train_data[name] = (group['sales'].values[-input_window_size:], group['time'].values[-input_window_size:])
    for name, group in test.groupby(['store', 'item']):
        group = group.copy(deep=False)
        group.index = range(len(group))
        group['time'] = group['date'].apply(lambda x: (_date(x) - start).days)
        predict_data[name] = (group['time'].values, group['id'].values)
    in_set = []
    ids = []
    for name, v1 in train_data.items():
        true_times, true_id = predict_data[name]
        if len(true_times) < output_window_size:
            true_times = np.pad(true_times, (0, output_window_size - len(true_times)), 'constant')
        in_set.append((v1[0], v1[0], true_times))
        ids.append(true_id)

    def _gen():
        for i in in_set:
            yield (i[0].astype(np.float32).reshape([-1, 1]),
                   np.array(0, dtype=np.float32),
                   i[1].astype(np.int64).reshape([-1, 1]),
                   np.array(0, dtype=np.float32),
                   i[2].astype(np.int64).reshape([-1, 1]))

    def _out_fn():
        tf_set = tf.data.Dataset.from_generator(lambda: _gen(),
                                                (
                                                    tf.float32, tf.float32, tf.int64, tf.float32,
                                                    tf.int64),
                                                (
                                                    [input_window_size, 1],
                                                    tf.TensorShape([]),
                                                    [input_window_size, 1],
                                                    tf.TensorShape([]),
                                                    [output_window_size, 1]))
        tf_set = tf_set.batch(1)
        return tf_set.map(_test_output_format)

    return ids, _out_fn


class CSVDataSet:
    def __init__(self, params, input_files=None):
        if input_files is None:
            inputs = glob.glob(params['data_set'])
        else:
            inputs = input_files
        self.input_window_size = params['input_window_size']
        self.output_window_size = params['output_window_size']
        self.window_length = params['output_window_size'] + params['input_window_size']
        self.exclude_columns = params['exclude_feature_columns']
        self.files = {}
        for file in inputs:
            row_count = sum(1 for _ in open(file))
            if row_count < self.window_length:
                continue
            self.files[file] = row_count
        tmp = next(pd.read_csv(next(iter(self.files.keys())), parse_dates=False, chunksize=1))
        self.features_index = []
        self.exogenous_index = []
        self.time_index = None
        self.cols_index = {}
        for i, c in enumerate(tmp.columns):
            if c in self.exclude_columns:
                continue
            self.cols_index[c] = i
            if c == params['timestamp_column']:
                self.time_index = i
            elif c in params['exogenous_feature_columns']:
                self.exogenous_index.append(i)
            else:
                self.features_index.append(i)
        logging.info('Exogenous Index: {}'.format(self.exogenous_index))
        logging.info('Features Index: {}'.format(self.features_index))
        logging.info('Timestamp Index: {}'.format(self.time_index))

    def gen(self, is_train, train_eval_split=False):
        logging.info("Use custom split on train and validation?: {}".format(train_eval_split))
        loop = itertools.count(1) if is_train else range(1)
        _exogenous = len(self.exogenous_index) > 0
        for _ in loop:
            for file, file_size in self.files.items():
                data = pd.read_csv(file, parse_dates=False)
                variables = data.iloc[:, self.features_index].as_matrix()
                exogenous = data.iloc[:, self.exogenous_index].as_matrix() if _exogenous else 0
                times = data.iloc[:, [self.time_index]].as_matrix()
                if train_eval_split and is_train:
                    variables = variables[0:-self.output_window_size]
                    times = times[0:-self.output_window_size]
                    if _exogenous:
                        exogenous = exogenous[0:-self.output_window_size]
                    file_size = max(0, file_size - self.output_window_size)
                elif train_eval_split:
                    variables = variables[-self.window_length:]
                    times = times[-self.window_length:]
                    if _exogenous:
                        exogenous = exogenous[-self.window_length:]
                    file_size = self.window_length
                offset = random.randint(0,
                                        min(2, file_size - self.window_length)) if is_train else 0
                for i in range(offset, variables.shape[0], self.window_length):
                    if i + self.window_length > variables.shape[0]:
                        break
                    else:
                        end = i + self.input_window_size
                    yield (variables[i:end].astype(np.float32),
                           exogenous[i:end].astype(np.float32) if _exogenous else np.array(0, dtype=np.float32),
                           times[i:end].astype(np.int64),
                           variables[end:end + self.output_window_size].astype(np.float32),
                           exogenous[end:end + self.output_window_size].astype(np.float32) if _exogenous else np.array(
                               0, dtype=np.float32),
                           times[end:end + self.output_window_size].astype(np.int64))

    def input_fn(self, is_train, batch_size, train_eval_split=False):
        def _out_fn():
            _exogenous_input_shape = [self.input_window_size, len(self.exogenous_index)] if len(
                self.exogenous_index) > 0 else tf.TensorShape([])
            _exogenous_output_shape = [self.output_window_size, len(self.exogenous_index)] if len(
                self.exogenous_index) > 0 else tf.TensorShape([])
            tf_set = tf.data.Dataset.from_generator(lambda: self.gen(is_train, train_eval_split=train_eval_split),
                                                    (
                                                        tf.float32, tf.float32, tf.int64, tf.float32, tf.float32,
                                                        tf.int64),
                                                    (
                                                        [self.input_window_size, len(self.features_index)],
                                                        _exogenous_input_shape,
                                                        [self.input_window_size, 1],
                                                        [self.output_window_size, len(self.features_index)],
                                                        _exogenous_output_shape,
                                                        [self.output_window_size, 1]))
            if is_train:
                tf_set = tf_set.batch(batch_size)
            else:
                tf_set = tf_set.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

            return tf_set.map(_train_output_format)

        return _out_fn


def encoder_model_fn(features, y_variables, mode, params=None, config=None):
    logging.info('Build Model')
    global_step = tf.train.get_or_create_global_step()
    x_variables, x_exogenous, x_times = features['inputs']
    y_exogenous, y_times = features['outputs']

    variables_mean, variables_var = tf.nn.moments(x_variables, axes=[1], keep_dims=True)
    x_variables = tf.nn.batch_normalization(x_variables, variables_mean, variables_var, None, None, 1e-3)

    _exogenous = len(x_exogenous.shape) > 1
    logging.info('Use Exogenous features: {}, shape: {}'.format(_exogenous, x_exogenous.shape))

    if _exogenous:
        exogenous_mean, exogenous_var = tf.nn.moments(x_exogenous, axes=[1], keep_dims=True)
        x_exogenous = tf.nn.batch_normalization(x_exogenous, exogenous_mean, exogenous_var, None, None, 1e-3)
        y_exogenous = tf.nn.batch_normalization(y_exogenous, exogenous_mean, exogenous_var, None, None, 1e-3)
        inputs = tf.concat([x_variables, x_exogenous], axis=-1)
    else:
        inputs = x_variables

    if params['time_periods'] is not None and len(params['time_periods']) > 0:
        x_times = _time_features(x_times, params['time_periods'], params['time_buckets'])
        y_times = _time_features(y_times, params['time_periods'], params['time_buckets'])
        inputs = tf.concat([inputs, x_times], axis=-1)
        if _exogenous:
            output = tf.concat([y_exogenous, y_times], axis=-1)
        else:
            output = y_times
    else:
        if _exogenous:
            output = y_exogenous
        else:
            output = tf.zeros([params['batch_size'], y_times.shape[1], 1], dtype=tf.float32)

    if params['input_layer'] == 'cnn':
        epsilon = 1e-3
        features_size = inputs.shape[2]
        inputs = tf.reshape(inputs, [params['batch_size'], -1, features_size, 1])
        cnn1 = tf.layers.conv2d(inputs, filters=32, kernel_size=[7, features_size], strides=[1, 1],
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same')
        batch_mean, batch_var = tf.nn.moments(cnn1, [0, 1, 2], shift=None, name="moments_cnn1", keep_dims=True)
        cnn1 = tf.nn.batch_normalization(cnn1, batch_mean, batch_var, None, None, epsilon, name="batch_norm_cnn1")
        ht1 = tf.minimum(20.0, tf.maximum(0.0, cnn1))
        cnn2 = tf.layers.conv2d(ht1, filters=32, kernel_size=[7, 1], strides=[1, 1],
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same')
        batch_mean, batch_var = tf.nn.moments(cnn2, [0, 1, 2], shift=None, name="moments_cnn2", keep_dims=True)
        cnn2 = tf.nn.batch_normalization(cnn2, batch_mean, batch_var, None, None, epsilon, name="batch_norm_cnn2")
        inputs = tf.minimum(20.0, tf.maximum(0.0, cnn2))
        inputs = tf.reshape(inputs, [params['batch_size'], -1, features_size * 32])

    inputs = tf.transpose(inputs, perm=[1, 0, 2])
    output = tf.transpose(output, perm=[1, 0, 2])

    if params['input_layer'] == 'cnn':
        rnn_inputs = inputs
    else:
        rnn_inputs = tf.layers.dense(inputs, params['hidden_size'],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

    if params['dropout'] is not None:
        rnn_inputs = tf.layers.dropout(inputs=rnn_inputs, rate=params['dropout'],
                                       training=mode == tf.estimator.ModeKeys.TRAIN)
    enc_output = rnn_inputs
    for _ in range(params['num_layers'] - 1):
        encoder = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'])
        enc_output, _ = encoder(enc_output, dtype=tf.float32)
    encoder = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'])
    decoder = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'])
    _, encoder_state = encoder(enc_output, dtype=tf.float32)
    decoder_output, _ = decoder(output, initial_state=encoder_state, dtype=tf.float32)
    decoder_output = tf.transpose(decoder_output, [1, 0, 2])
    rnn_outputs = tf.layers.dense(decoder_output, x_variables.shape[2],
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())

    metrics = {}
    predictions = rnn_outputs / tf.rsqrt(variables_var + 1e-3) + variables_mean
    predictions = tf.round(predictions)
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        #labels = tf.nn.batch_normalization(y_variables, variables_mean, variables_var, None, None, 1e-3)
        loss_op = tf.losses.mean_squared_error(y_variables, predictions)
        denominator = tf.abs(predictions) + tf.abs(y_variables)
        denominator = tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)
        smape = 200 * tf.abs(predictions - y_variables) / denominator
        metrics['SMAPE'] = tf.metrics.mean(smape)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('SMAPE', tf.reduce_mean(smape))

    else:
        loss_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        gvs = opt.compute_gradients(loss_op)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = opt.apply_gradients(capped_gvs, global_step=global_step)
    else:
        train_op = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        predictions = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=predictions,
        loss=loss_op,
        train_op=train_op)


def _time_features(time, periods, buckets):
    batch_size = tf.shape(time)[0]
    num_periods = len(periods)
    periods = tf.constant(periods, shape=[1, 1, num_periods, 1], dtype=time.dtype)
    time = tf.reshape(time, [batch_size, -1, 1, 1])
    mod = (tf.cast(time % periods, tf.float32) * buckets / tf.cast(periods, tf.float32))
    intervals = tf.reshape(tf.range(buckets, dtype=tf.float32), [1, 1, 1, buckets])
    mod = tf.nn.relu(mod - intervals)
    mod = tf.where(mod < 1.0, mod, tf.zeros_like(mod))
    mod = tf.reshape(mod, [batch_size, -1, num_periods * buckets])
    return mod


class CSVTimeSeriesModel(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return encoder_model_fn(
                features=features,
                y_variables=labels,
                mode=mode,
                params=params,
                config=config)

        super(CSVTimeSeriesModel, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )


class BestExporter(tf.estimator.Exporter):
    def __init__(self, name, keep_max=3):
        self._name = name
        self._best = None
        self._keep_max = keep_max
        logging.info("Setup base exporter")

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        if not tf.gfile.Exists(export_path):
            tf.gfile.MakeDirs(export_path)
        global_step = eval_result['global_step']
        results = export_path + "/best.csv"
        entry = pd.DataFrame(data={
            'SMAPE': [eval_result['SMAPE']],
            'loss': [eval_result['loss']],
            'global_step': [global_step],
            'checkpoint': [checkpoint_path]
        })
        if self._best is None:
            if tf.gfile.Exists(results):
                self._best = pd.read_csv(results, parse_dates=False)
                self._best = self._best.append(entry)
            else:
                self._best = entry
        else:
            self._best = self._best.append(entry)

        self._best.sort_values(by=['SMAPE'], ascending=True, inplace=True)
        self._best = self._best[0:min(self._keep_max, len(self._best))]
        self._best.to_csv(results, header=True, index=False)

        steps = list(self._best['global_step'])
        if global_step in steps:
            logging.info('Copy {} checkpoint to best one.'.format(global_step))
            for mf in glob.iglob(checkpoint_path + '.*'):
                name = os.path.basename(mf)
                shutil.copyfile(mf, os.path.join(export_path, name))
        for mf in glob.iglob(export_path + '/model.ckpt-*'):
            logging.info('Found: {}'.format(mf))
            name = os.path.basename(mf)
            name = name.lstrip('model.ckpt-')
            p = name.split('.')
            logging.info('Check: {} in {}'.format(p, steps))
            if len(p) > 1:
                s = int(p[0])
                if s not in steps:
                    logging.info('Drop checkpoint file: {}'.format(mf))
                    os.remove(mf)
