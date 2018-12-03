"""
First, you need to compile the Bouma baseline:
>>> gcc -o hyphen hyphenate_mnl.c
"""

import subprocess
import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import Levenshtein

import syllabification.utils as u

def main():

    def eval(silver_file, gold_file):
        silver = u.load_file(silver_file)
        _, silver_y = u.x_and_y(silver)

        gold = u.load_file(gold_file)
        _, gold_y = u.x_and_y(gold)
        
        acc_syll = accuracy_score([i for s in gold_y for i in s],
                                [i for s in silver_y for i in s])
        f1_syll = f1_score([i for s in gold_y for i in s],
                                [i for s in silver_y for i in s])
        acc_token = accuracy_score([str(s) for s in gold_y], 
                                        [str(s) for s in silver_y])

        silver_tokens = [l.strip() for l in open(silver_file)]
        gold_tokens = [l.strip() for l in open(gold_file)]

        lev = np.mean([Levenshtein.distance(g, s) for g, s in zip(gold_tokens, silver_tokens)])
            
        return acc_syll, f1_syll, acc_token, lev
    
    print('Bouma et al. baseline:')
    acc_syll, f1_syll, acc_token, lev = eval('model_b/bouma_dev.txt', 'data/splits/dev.txt')
    print('- dev scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    acc_syll, f1_syll, acc_token, lev = eval('model_b/bouma_test.txt', 'data/splits/test.txt')
    print('- test scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    print('Plain CRF baseline:')
    acc_syll, f1_syll, acc_token, lev = eval('model_b/silver_dev.txt', 'data/splits/dev.txt')
    print('- dev scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    acc_syll, f1_syll, acc_token, lev = eval('model_b/silver_test.txt', 'data/splits/test.txt')
    print('- test scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    print('Our system (LSTM + CRF):')
    acc_syll, f1_syll, acc_token, lev = eval('model_s/silver_dev.txt', 'data/splits/dev.txt')
    print('- dev scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    acc_syll, f1_syll, acc_token, lev = eval('model_s/silver_test.txt', 'data/splits/test.txt')
    print('- test scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)
    
if __name__ == '__main__':
    main()