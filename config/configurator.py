import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default 4, help='number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256, help='embedding size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--block_size', type=int, default=64, help='block size')
    parser.add_argument('--max_iters', type=int, default=10000, help='max iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='evaluation iterations')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    return parser.parse_args()

if __name__ == '__main__':
    config = get_config()
    print(config)
