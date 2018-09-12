import tensorflow as tf
import argparse
import os
import logging
import configparser
import models.csv as fcsv
import json
import pandas as pd
import numpy as np


def parse_args():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs',
    )
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument(
        '--checkpoint_path',
        default="",
        help='Checkpoint path',
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='Number RNN layers.',
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='LSTM hidden size.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size.',
    )
    parser.add_argument(
        '--input_window_size',
        type=int,
        default=90,
        help='input_window_size',
    )
    parser.add_argument(
        '--output_window_size',
        type=int,
        default=30,
        help='output_window_size',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Dropout',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Recommended learning_rate is 0.0003.',
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=None,
        help='Norm for gradients clipping.',
    )
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=10,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=None,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=5,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=100,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--exogenous_feature_columns',
        default="",
        help='Exogenous column names. Comma separated',
    )
    parser.add_argument(
        '--timestamp_column',
        default="time",
        help='Timestamp column name',
    )
    parser.add_argument(
        '--exclude_feature_columns',
        default="",
        help='Exclude column names. Comma separated',
    )
    parser.add_argument(
        '--timestamp_column_format',
        default="",
        help='Timestamp column format',
    )
    parser.add_argument(
        '--data_set',
        default=None,
        help='Location of training files or evaluation files',
    )
    parser.add_argument(
        '--time_periods',
        default=None,
        help='Time periods',
    )
    parser.add_argument(
        '--time_buckets',
        default=10,
        type=int,
        help='Time buckets',
    )
    parser.add_argument(
        '--input_layer',
        default='dense',
        help='Input layer',
    )

    parser.add_argument(
        '--train_eval_split',
        dest='train_eval_split',
        action='store_true',
        help='Split train data to train and eval',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    p_file = os.path.join(checkpoint_dir, 'parameters.ini')
    if tf.gfile.Exists(p_file):
        parameters = configparser.ConfigParser(allow_no_value=True)
        parameters.read(p_file)
        parser.set_defaults(**dict(parameters.items("PARAMETERS", raw=True)))
    args = parser.parse_args(remaining_argv)
    print('\n*************************\n')
    print(args)
    print('\n*************************\n')
    return checkpoint_dir, args


def make_list_variable(params, name):
    if params[name] is None or params[name] == '':
        params[name] = []
    else:
        params[name] = params[name].split(',')


def test(checkpoint_dir, checkpoint_path, params):
    logging.info("start test  model")

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
    )
    make_list_variable(params, 'exclude_feature_columns')
    make_list_variable(params, 'exogenous_feature_columns')
    make_list_variable(params, 'time_periods')
    params['time_periods'] = [int(s) for s in params['time_periods']]

    lstm = fcsv.CSVTimeSeriesModel(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )

    params['batch_size'] = 1
    ids, data_fn = fcsv.submit_input_fn(params['data_set'] + '/train.csv', params['data_set'] + '/test.csv',
                                        params['input_window_size'],params['output_window_size'])
    predictions = lstm.predict(input_fn=data_fn, checkpoint_path=checkpoint_path)
    id = []
    value = []
    j = 0
    for p in predictions:
        pid = ids[j]
        for i in range(len(pid)):
            value.append(int(p[i]))
            id.append(pid[i])
        j+=1
    submission = pd.DataFrame({'id': id, 'sales': value})
    submission.sort_values(by=['id'], ascending=True, inplace=True)
    submission.to_csv(checkpoint_dir + '/submission.csv', header=True, index=False)
    true_data = pd.read_csv(params['data_set'] + '/submit.csv')
    true_data.sort_values(by=['id'], ascending=True, inplace=True)
    check_id = ((submission['id'].values-true_data['id'].values).mean() == 0)
    logging.info('IDS check ok: {}'.format(check_id))

    d = abs(submission['sales'].values)+abs(true_data['sales'].values)
    d[d == 0] = 1
    smape = 200 * (abs(submission['sales'].values - true_data['sales'].values) / d).sum()/len(submission)
    logging.info('SMAPE: {}'.format(smape))




def train(mode, checkpoint_dir, train_eval_split, params):
    logging.info("start build  model")

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']
    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps']
    )
    make_list_variable(params, 'exclude_feature_columns')
    make_list_variable(params, 'exogenous_feature_columns')
    make_list_variable(params, 'time_periods')
    params['time_periods'] = [int(s) for s in params['time_periods']]

    lstm = fcsv.CSVTimeSeriesModel(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    logging.info("Start %s mode", mode)
    data = fcsv.CSVDataSet(params)
    if mode == 'train':
        lstm.train(input_fn=data.input_fn(True, params['batch_size'], train_eval_split=train_eval_split))
    else:
        train_fn = fcsv.null_dataset()
        train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
        eval_fn = data.input_fn(False, params['batch_size'], train_eval_split=train_eval_split)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=None, start_delay_secs=10, throttle_secs=60,
                                          exporters=fcsv.BestExporter(name='best_results'))
        tf.estimator.train_and_evaluate(lstm, train_spec, eval_spec)


def main():
    checkpoint_dir, args = parse_args()
    logging.info('------------------')
    logging.info('TF VERSION: {}'.format(tf.__version__))
    logging.info('ARGS: {}'.format(args))
    logging.info('------------------')
    if args.worker:
        mode = 'train'
    elif args.test:
        mode = 'test'
    else:
        mode = 'eval'
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })

    params = {
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'input_window_size': args.input_window_size,
        'output_window_size': args.output_window_size,
        'learning_rate': args.learning_rate,
        'grad_clip': args.grad_clip,
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'timestamp_column': args.timestamp_column,
        'exogenous_feature_columns': args.exogenous_feature_columns,
        'exclude_feature_columns': args.exclude_feature_columns,
        'data_set': args.data_set,
        'time_periods': args.time_periods,
        'time_buckets': args.time_buckets,
        'input_layer': args.input_layer,
        'train_eval_split': args.train_eval_split,
        'dropout': args.dropout
    }

    if args.test:
        test(checkpoint_dir,args.checkpoint_path,params)
        return

    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    train(mode, checkpoint_dir, args.train_eval_split, params)


if __name__ == '__main__':
    main()
