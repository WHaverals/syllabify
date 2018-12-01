"""
First, you need to compile:
>>> gcc -o hyphen hyphenate_mnl.c
"""

import subprocess
import argparse
import os

from sklearn.metrics import accuracy_score, f1_score

import syllabification.utils as u

def main():
    parser = argparse.ArgumentParser(description='Applies Bouma et al. syllabifier')
    parser.add_argument('--input_dir', type=str,
                        default='data/splits/',
                        help='location of the splits folder')
    parser.add_argument('--output_dir', type=str,
                        default='model_s',
                        help='location of the model folder')
    args = parser.parse_args()
    print(args)

    with open('tmp.txt', 'w') as f:
        for line in open(f'{args.input_dir}/dev.txt'):
            f.write(line.replace('-', ''))

    cmd = f'./boumaEtAl/hyphen < tmp.txt > {args.output_dir}/bouma_dev.txt'
    subprocess.call(cmd, shell=True)

    with open('tmp.txt', 'w') as f:
        for line in open(f'{args.input_dir}/test.txt'):
            f.write(line.replace('-', ''))

    cmd = f'./boumaEtAl/hyphen < tmp.txt > {args.output_dir}/bouma_test.txt'
    subprocess.call(cmd, shell=True)

    os.remove('tmp.txt')

    print('Bouma et al. baseline:')
    print('- dev scores:')
    silver = u.load_file(f'{args.output_dir}/bouma_dev.txt')
    _, silver_y = u.x_and_y(silver)

    gold = u.load_file(f'{args.input_dir}/dev.txt')
    _, gold_y = u.x_and_y(gold)
    
    acc_syll = accuracy_score([i for s in gold_y for i in s],
                              [i for s in silver_y for i in s])
    f1_syll = f1_score([i for s in gold_y for i in s],
                             [i for s in silver_y for i in s])
    acc_token = accuracy_score([str(s) for s in gold_y], 
                                    [str(s) for s in silver_y])
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)

    print('- test scores:')
    silver = u.load_file(f'{args.output_dir}/bouma_test.txt')
    _, silver_y = u.x_and_y(silver)

    gold = u.load_file(f'{args.input_dir}/test.txt')
    _, gold_y = u.x_and_y(gold)
    
    acc_syll = accuracy_score([i for s in gold_y for i in s],
                                   [i for s in silver_y for i in s])
    f1_syll = f1_score([i for s in gold_y for i in s],
                             [i for s in silver_y for i in s])
    acc_token = accuracy_score([str(s) for s in gold_y], 
                                    [str(s) for s in silver_y])
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)

    print('Our system:')
    silver = u.load_file(f'{args.output_dir}/silver_dev.txt')
    _, silver_y = u.x_and_y(silver)

    gold = u.load_file(f'{args.input_dir}/dev.txt')
    _, gold_y = u.x_and_y(gold)
    
    acc_syll = accuracy_score([i for s in gold_y for i in s],
                                   [i for s in silver_y for i in s])
    f1_syll = f1_score([i for s in gold_y for i in s],
                             [i for s in silver_y for i in s])
    acc_token = accuracy_score([str(s) for s in gold_y], 
                                    [str(s) for s in silver_y])
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)

    print('- test scores:')
    silver = u.load_file(f'{args.output_dir}/silver_test.txt')
    _, silver_y = u.x_and_y(silver)

    gold = u.load_file(f'{args.input_dir}/test.txt')
    _, gold_y = u.x_and_y(gold)
    
    acc_syll = accuracy_score([i for s in gold_y for i in s],
                                   [i for s in silver_y for i in s])
    f1_syll = f1_score([i for s in gold_y for i in s],
                             [i for s in silver_y for i in s])
    acc_token = accuracy_score([str(s) for s in gold_y], 
                                    [str(s) for s in silver_y])
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)

if __name__ == '__main__':
    main()