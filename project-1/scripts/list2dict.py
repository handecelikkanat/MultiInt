import sys
from argparse import ArgumentParser
import pickle


parser = ArgumentParser(description='Converts representations-list to representations-dict')
parser.add_argument('-list_file',
                    type=str)
parser.add_argument('-bpe_file',
                    type=str)
parser.add_argument('-dict_file',
                    type=str)
args = parser.parse_args()

reprs_list = pickle.load(open(args.list_file, 'rb'))
bpe_lines = open(args.bpe_file, 'r').readlines()

reprs_dict = dict()
for sent_reprs, sent_bpe in zip(reprs_list, bpe_lines):
    sent_key = sent_bpe.split('\t')[2]
    reprs_dict[sent_key] = sent_reprs

pickle.dump(reprs_dict, open(args.dict_file, 'wb'))
