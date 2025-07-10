import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training settings for model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--window', type=int, default=3, help='Input sequence length')  # 5 for Debutanizer
    parser.add_argument('--length_size', type=int, default=1, help='Prediction sequence length')
    parser.add_argument('--early_patience', type=float, default=0.5, help='Early stopping patience ratio')
    parser.add_argument('--scheduler_patience', type=float, default=0.25, help='LR scheduler patience ratio')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='Feed-forward dimension')
    parser.add_argument('--factor', type=int, default=5, help='Attention factor')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--Debutanizer_path', type=str, default='data\\Debutanizer.csv', help='path of data1')
    parser.add_argument('--SRU_path', type=str, default='data\\SRU.csv', help='path of data2')
    parser.add_argument('--dataset', type=str, default='Debutanizer', choices=['SRU','Debutanizer'])

    return parser.parse_args()