import tensorflow as tf
import itertools
import random
import numpy as np
import logging
import glob
import pandas as pd
import shutil
import os
from mlboardclient.api import client


def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def _train_output_format(x_variables, x_exogenous, x_times, y_variables, y_exogenous, y_times):
    return {'inputs': (x_variables, x_exogenous, x_times), 'outputs': (y_exogenous, y_times)}, y_variables


def _test_output_format(x_variables, x_exogenous, x_times, y_exogenous, y_times):
    return {'inputs': (x_variables, x_exogenous, x_times), 'outputs': (y_exogenous, y_times)}


def submit_input_fn(train, test, params):
    exogenous_columns = []
    for c in ['day']:
        exogenous_columns.append(c)

    if params['weekday_bucket']:
        for i in range(7):
            exogenous_columns.append('w{}'.format(i))
    else:
        exogenous_columns.append('weekday')

    if params['month_bucket']:
        for i in range(12):
            exogenous_columns.append('m{}'.format(i))
    else:
        exogenous_columns.append('month')
    if params['quoter_bucket']:
        for i in range(3):
            exogenous_columns.append('q{}'.format(i))
    else:
        exogenous_columns.append('quoter')

    def _extend(f):
        f['date'] = f['date'].apply(lambda x: pd.Timestamp(x))
        f['day'] = f['date'].apply(lambda x: x.day)
        # f['year'] = f['date'].apply(lambda x: x.year)
        if params['weekday_bucket']:
            for i in range(7):
                j = i
                c = 'w{}'.format(i)
                f[c] = f['date'].apply(lambda x: 1 if x.weekday() == j else 0)
        else:
            f['weekday'] = f['date'].apply(lambda x: x.weekday())

        if params['month_bucket']:
            for i in range(12):
                j = i + 1
                c = 'm{}'.format(i)
                f[c] = f['date'].apply(lambda x: 1 if x.month == j else 0)
        else:
            f['month'] = f['date'].apply(lambda x: x.month)

        if params['quoter_bucket']:
            for i in range(3):
                j = i
                c = 'q{}'.format(i)
                f[c] = f['date'].apply(lambda x: 1 if ((x.month - 1) % 3) == j else 0)
        else:
            f['quoter'] = f['date'].apply(lambda x: ((x.month - 1) % 3))
        f.set_index('date', inplace=True)
        return f

    train = _extend(pd.read_csv(train, parse_dates=False))
    test = _extend(pd.read_csv(test, parse_dates=False))

    train_data = {}
    predict_data = {}
    train_groups = {}
    for name, group in train.groupby(['store', 'item']):
        values = group['sales'].values[-params['input_window_size']:]
        exogenous = group.loc[:, exogenous_columns].values[-params['input_window_size']:, :]
        times = np.zeros(values.shape, dtype=np.int64)
        for i in [4, 6, 12]:
            t = group.reindex(group.index[-params['input_window_size']:] - pd.DateOffset(months=i))
            t.fillna(inplace=True, value=-1)
            lags = t.loc[:, ['sales']].values
            exogenous = np.concatenate((exogenous, lags), axis=-1)
        train_groups[name] = group
        train_data[name] = (values,
                            exogenous,
                            times)
    for name, group in test.groupby(['store', 'item']):
        exogenous = group.loc[:, exogenous_columns].values
        ids = group['id'].values
        times = np.zeros(len(ids), dtype=np.int64)
        tgroup = train_groups[name]
        for i in [4, 6, 12]:
            t = tgroup.reindex(group.index - pd.DateOffset(months=i))
            t.fillna(inplace=True, value=-1)
            lags = t.loc[:, ['sales']].values
            exogenous = np.concatenate((exogenous, lags), axis=-1)
        predict_data[name] = (
            exogenous,
            times,
            ids)
    in_set = []
    ids = []
    for name, v1 in train_data.items():
        true_ext, true_times, true_id = predict_data[name]
        if len(true_times) < params['output_window_size']:
            true_times = np.pad(true_times, (0, params['output_window_size'] - len(true_times)), 'constant')
            true_ext = np.pad(true_ext, ((0, params['output_window_size'] - len(true_ext)), (0, 0)), 'constant')
        in_set.append((v1[0], v1[1], v1[2], true_ext, true_times))
        ids.append(true_id)

    def _gen():
        for i in in_set:
            yield (i[0].astype(np.float32).reshape([-1, 1]),
                   i[1].astype(np.float32),
                   i[2].astype(np.int64).reshape([-1, 1]),
                   i[3].astype(np.float32),
                   i[4].astype(np.int64).reshape([-1, 1]))

    def _out_fn():
        tf_set = tf.data.Dataset.from_generator(lambda: _gen(),
                                                (
                                                    tf.float32, tf.float32, tf.int64, tf.float32,
                                                    tf.int64),
                                                (
                                                    [params['input_window_size'], 1],
                                                    [params['input_window_size'], len(exogenous_columns) + 3],
                                                    [params['input_window_size'], 1],
                                                    [params['output_window_size'], len(exogenous_columns) + 3],
                                                    [params['output_window_size'], 1]))
        tf_set = tf_set.batch(1)
        return tf_set.map(_test_output_format)

    return ids, _out_fn


class CSVDataSet:
    def __init__(self, params, input_files=None):
        logging.info("TEST version: 1")
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
            data = pd.read_csv(file, parse_dates=False)
            # row_count = sum(1 for _ in open(file))
            row_count = len(data)
            if row_count < self.window_length:
                continue
            if (params['threshold']) is not None and (data['sales'].mean() > params['threshold']):
                continue
            if (params['threshold1']) is not None and (data['sales'].mean() <= params['threshold1']):
                continue
            self.files[file] = row_count

        tmp = next(pd.read_csv(next(iter(self.files.keys())), parse_dates=False, chunksize=1))
        self.features_columns = []
        self.exogenous_columns = []
        self.time_column = None
        self.cols_index = {}
        for i, c in enumerate(tmp.columns):
            if c in self.exclude_columns:
                continue
            self.cols_index[c] = i
            if c == params['timestamp_column']:
                self.time_column = c
            elif c in params['exogenous_feature_columns']:
                self.exogenous_columns.append(c)
            else:
                self.features_columns.append(c)
        for c in ['day']:
            self.exogenous_columns.append(c)

        # params['weekday_bucket'] = False
        if params['weekday_bucket']:
            for i in range(7):
                self.exogenous_columns.append('w{}'.format(i))
        else:
            self.exogenous_columns.append('weekday')

        if params['month_bucket']:
            for i in range(12):
                self.exogenous_columns.append('m{}'.format(i))
        else:
            self.exogenous_columns.append('month')
        if params['quoter_bucket']:
            for i in range(3):
                self.exogenous_columns.append('q{}'.format(i))
        else:
            self.exogenous_columns.append('quoter')

        logging.info('Exogenous Index: {}'.format(self.exogenous_columns))
        logging.info('Features Index: {}'.format(self.features_columns))
        logging.info('Timestamp Index: {}'.format(self.time_column))
        self._buffer = {}
        self._params = params

    def gen(self, is_train, train_eval_split=False):
        logging.info("Use custom split on train and validation?: {}".format(train_eval_split))
        loop = itertools.count(1) if is_train else range(1)
        _exogenous = len(self.exogenous_columns) > 0

        def from_buffer(file):
            v = self._buffer.get(file, None)
            if v is None:
                data = pd.read_csv(file, parse_dates=False)
                data['date'] = data['date'].apply(lambda x: pd.Timestamp(x))
                item = data.loc[:, 'item'].values[0]
                store = data.loc[:, 'store'].values[0]

                if self._params['weekday_bucket']:
                    for i in range(7):
                        j = i
                        c = 'w{}'.format(i)
                        data[c] = data['date'].apply(lambda x: 1 if x.weekday() == j else 0)
                else:
                    data['weekday'] = data['date'].apply(lambda x: x.weekday())

                if self._params['month_bucket']:
                    for i in range(12):
                        j = i + 1
                        c = 'm{}'.format(i)
                        data[c] = data['date'].apply(lambda x: 1 if x.month == j else 0)
                else:
                    data['month'] = data['date'].apply(lambda x: x.month)

                if self._params['quoter_bucket']:
                    for i in range(3):
                        j = i
                        c = 'q{}'.format(i)
                        data[c] = data['date'].apply(lambda x: 1 if ((x.month - 1) % 3) == j else 0)
                else:
                    data['quoter'] = data['date'].apply(lambda x: ((x.month - 1) % 3))

                data['day'] = data['date'].apply(lambda x: x.day)
                # data['year'] = data['date'].apply(lambda x: x.year)

                data.set_index('date', inplace=True)

                variables = data.loc[:, self.features_columns].values
                exogenous = data.loc[:, self.exogenous_columns].values if _exogenous else 0
                times = data.loc[:, [self.time_column]].values
                data = data[self.features_columns]
                for i in [4, 6, 12]:
                    t = data.reindex(data.index - pd.DateOffset(months=i))
                    t.fillna(inplace=True, value=-1)
                    lags = t.loc[:, self.features_columns].values
                    exogenous = np.concatenate((exogenous, lags), axis=-1)
                v = (item, store, variables, exogenous, times)
                self._buffer[file] = v
                return v
            else:
                return v

        epoch = 0
        for _ in loop:
            logging.info('epoch: {}'.format(epoch))
            epoch += 1
            for file, file_size in self.files.items():
                item, store, variables, exogenous, times = from_buffer(file)
                index = item + store - 2
                # if is_train and ((index % 4) == 0):
                #    continue
                # elif (not is_train) and ((index % 4) != 0):
                #    continue
                if train_eval_split and is_train:
                    variables = variables[0:-self.window_length]
                    times = times[0:-self.window_length]
                    if _exogenous:
                        exogenous = exogenous[0:-self.window_length]
                    file_size = max(0, file_size - self.window_length)
                elif train_eval_split:
                    variables = variables[-self.window_length:]
                    times = times[-self.window_length:]
                    if _exogenous:
                        exogenous = exogenous[-self.window_length:]
                    file_size = self.window_length
                offset = random.randint(0,
                                        min(self.input_window_size, file_size - self.window_length)) if is_train else 0
                for i in range(offset, variables.shape[0], self.window_length):
                    if i + self.window_length > variables.shape[0]:
                        continue
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
            _exogenous_input_shape = [self.input_window_size,
                                      len(self.exogenous_columns) + len(self.features_columns) * 3] if len(
                self.exogenous_columns) > 0 else tf.TensorShape([])
            _exogenous_output_shape = [self.output_window_size,
                                       len(self.exogenous_columns) + len(self.features_columns) * 3] if len(
                self.exogenous_columns) > 0 else tf.TensorShape([])
            tf_set = tf.data.Dataset.from_generator(lambda: self.gen(is_train, train_eval_split=train_eval_split),
                                                    (
                                                        tf.float32, tf.float32, tf.int64, tf.float32, tf.float32,
                                                        tf.int64),
                                                    (
                                                        [self.input_window_size, len(self.features_columns)],
                                                        _exogenous_input_shape,
                                                        [self.input_window_size, 1],
                                                        [self.output_window_size, len(self.features_columns)],
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

    features_size = inputs.shape[2]
    #inputs = tf.reshape(inputs, [params['batch_size'], -1, features_size, 1])
    #inputs = tf.nn.avg_pool(inputs,[1,2,1,1],[1,1,1,1],'SAME')
    #inputs = tf.reshape(inputs, [params['batch_size'], -1, features_size])

    if params['input_layer'] == 'cnn':
        inputs = tf.reshape(inputs, [params['batch_size'], -1, features_size, 1])
        cnn1 = tf.layers.conv2d(inputs, filters=32, kernel_size=[7, features_size], strides=[1, 1],
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same')
        inputs = tf.tanh(cnn1)
        inputs = tf.reshape(inputs, [params['batch_size'], -1, features_size * 32])

    inputs = tf.transpose(inputs, perm=[1, 0, 2])
    output = tf.transpose(output, perm=[1, 0, 2])

    if params['input_layer'] == 'cnn':
        rnn_inputs = inputs
    else:
        rnn_inputs = tf.layers.dense(inputs, params['hidden_size'],activation=tf.nn.selu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())


    enc_output = rnn_inputs
    shape = [params['batch_size'], params['hidden_size']]
    for _ in range(params['num_layers'] - 1):
        encoder = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'])
        enc_output, _ = encoder(enc_output,dtype=tf.float32)
    encoder = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'])
    decoder = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'])
    _, encoder_state = encoder(enc_output,dtype=tf.float32)
    encoder_state = tf.contrib.rnn.LSTMStateTuple(
        tf.reshape(encoder_state[0],shape),
        tf.reshape(encoder_state[1],shape),
    )
    decoders = [decoder]
    states = [encoder_state]
    logging.info("Enc state {}".format(encoder_state))
    for _ in range(params['num_layers'] - 1):
        decoders.append(tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size']))
        states.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(tf.zeros(shape, tf.float32), trainable=False,collections=[tf.GraphKeys.LOCAL_VARIABLES]),
            tf.Variable(tf.zeros(shape, tf.float32), trainable=False,collections=[tf.GraphKeys.LOCAL_VARIABLES])))
    encoder_state =tuple(states)

    def cond_fn(time, prev_output, prev_state, targets):
        return time < params['output_window_size']

    def loop_fn(time, prev_output, prev_state, targets):
        next_input = tf.concat([prev_output, output[time:time + 1, :, :]], axis=-1)
        next_input = tf.layers.dense(next_input, params['hidden_size'],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        result = next_input
        newstates = []
        for i in range(params['num_layers']):
            result, state = decoders[i](result, initial_state=prev_state[i], dtype=tf.float32)
            newstates.append(state)

        if (params['dropout'] is not None) and (mode == tf.estimator.ModeKeys.TRAIN):
            result = tf.layers.dropout(inputs=result, rate=params['dropout'],
                                       training=mode == tf.estimator.ModeKeys.TRAIN)

        result = tf.layers.dense(result, x_variables.shape[2],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

        targets = targets.write(time, result[0])
        next_output = tf.concat([prev_output[:, :, x_variables.shape[2]:], result], axis=-1)
        return time + 1, next_output, tuple(newstates), targets

    back = x_variables[:, -params['look_back']:, :]
    back = tf.reshape(back, [1, params['batch_size'], params['look_back'] * x_variables.shape[2]])
    loop_init = [tf.constant(0, dtype=tf.int32), back,
                 encoder_state,
                 tf.TensorArray(dtype=tf.float32, size=params['output_window_size'])]

    _, _, _, decoder_output = tf.while_loop(cond_fn, loop_fn, loop_vars=loop_init)

    decoder_output = decoder_output.stack()
    rnn_outputs = tf.transpose(decoder_output, [1, 0, 2])

    metrics = {}
    predictions = rnn_outputs / tf.rsqrt(variables_var + 1e-3) + variables_mean
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        #denominator_loss = tf.abs(predictions) + tf.abs(y_variables) + 0.1
        #smape_loss = tf.abs(predictions - y_variables) / denominator_loss
        loss_op = tf.losses.mean_squared_error(y_variables,predictions)
        predictions = tf.round(predictions)
        denominator = tf.abs(predictions) + tf.abs(y_variables)
        denominator = tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)
        smape = tf.abs(predictions - y_variables) / denominator
        smape = 200 * smape
        metrics['SMAPE'] = tf.metrics.mean(smape)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('SMAPE', tf.reduce_mean(smape))
    else:
        loss_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        gvs = opt.compute_gradients(loss_op)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        # capped_gvs = gvs
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

        client.update_task_info({'SMAPE': float(eval_result['SMAPE']),
                                 'BEST_SMAPE': float(self._best['SMAPE'].values[0]),
                                 'BEST_STEP': int(self._best['global_step'].values[0]),
                                 })
        steps = list(self._best['global_step'])
        if global_step in steps:
            for mf in glob.iglob(checkpoint_path + '.*'):
                name = os.path.basename(mf)
                dest = os.path.join(export_path, name)
                if not tf.gfile.Exists(dest):
                    shutil.copyfile(mf, dest)

        for mf in glob.iglob(export_path + '/model.ckpt-*'):
            name = os.path.basename(mf)
            name = name.lstrip('model.ckpt-')
            p = name.split('.')
            if len(p) > 1:
                s = int(p[0])
                if s not in steps:
                    os.remove(mf)
