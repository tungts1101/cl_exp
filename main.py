import pandas as pd
import argparse
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    parser.add_argument('-d', type=str)
    parser.add_argument('-m', type=int, default=10000)
    a=parser.parse_args()
    exps=pd.read_csv('./args/'+a.d+'_publish.csv')
    args=exps[exps['ID']==a.i].to_dict('records')[0]
    args['seed']=[args['seed']]
    args['device']=[args['device']]
    args['do_not_save']=False
    args['M']=a.m
    train(args)

if __name__ == '__main__':
    main()