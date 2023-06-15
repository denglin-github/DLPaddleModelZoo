import argparse
import numpy as np
import sys

sys.path.insert(0, sys.path[0]+"/../")
from engine.dlnne_engine import Predictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--input_data", type=str, help="data filename")
    parser.add_argument("--disable_graphs", type=str, default='', help="disable graphs into cpu")
    parser.add_argument("--disable_nodes", type=str, default='', help="disable nodes into cpu")    
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()


def create_predictor():
    args = parse_args()
    predictor = Predictor(args.model_file,
                          args.params_file,
                          args.disable_graphs,
                          args.disable_nodes)
    return predictor

def main():
    args = parse_args()
    dlnne_predictor = create_predictor()
    input_data = np.load(args.input_data)
    result = dlnne_predictor.run([input_data])
    return result

if __name__ == "__main__":
    main()
    
    