import argparse
import tensorflow as tf

PARSER = argparse.ArgumentParser(description="UNet-Sentinel3")

PARSER.add_argument('--exec_mode',
                    choices=['train', 'train_and_predict', 'predict'],
                    type=str,
                    default='train_and_predict',
                    help="""Which execution mode to run the model into"""
                    )

PARSER.add_argument('--model_dir',
                    type=str,
                    default='./results',
                    help="""Output directory for information related to the model"""
                    )

PARSER.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="""Input directory containing the dataset for training the model"""
                    )

PARSER.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="""Size of each minibatch per GPU""")

PARSER.add_argument('--epochs',
                    type=int,
                    default=1000,
                    help="""Maximum number of steps (batches) used for training""")

PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help="""Random seed""")

PARSER.add_argument('--learning_rate',
                    type=float,
                    default=0.01,
                    help="""Learning rate coefficient for SGD""")

PARSER.add_argument('--use_amp', dest='use_amp', action='store_true',
                    help="""Train using TF-AMP""")
PARSER.set_defaults(use_amp=False)

PARSER.add_argument('--use_trt', dest='use_trt', action='store_true',
                    help="""Use TF-TRT""")
PARSER.set_defaults(use_trt=False)


def _cmd_params(flags):
    return {
        'model_dir': flags.model_dir,
        'batch_size': flags.batch_size,
        'data_dir': flags.data_dir,
        'epochs': flags.epochs,
        'dtype': tf.float32,
        'learning_rate': flags.learning_rate,
        'exec_mode': flags.exec_mode,
        'seed': flags.seed,
        'use_amp': flags.use_amp,
        'use_trt': flags.use_trt,
    }
