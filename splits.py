import argparse
import shutil
import os
from collections import Counter

from sklearn.model_selection import train_test_split as split

def main():
    parser = argparse.ArgumentParser(description='Splits available data in train-dev-test')
    parser.add_argument('--input_file', type=str,
                        default='data/syllabified_crm.txt',
                        help='location of the full data file')
    parser.add_argument('--split_dir', type=str,
                        default='data/splits',
                        help='location of the train-dev-test files')
    parser.add_argument('--train_prop', type=float,
                        default=.8,
                        help='Proportion of training items (dev and test are equal-size)')
    parser.add_argument('--seed', type=int,
                        default=43438,
                        help='Random seed')
    args = parser.parse_args()
    print(args)

    try:
        shutil.rmtree(args.split_dir)
    except FileNotFoundError:
        pass
    os.mkdir(args.split_dir)

    with open(args.input_file, 'r') as f:
        items = [l.strip() for l in f if l.strip()]
    
    num_sylls = [w.count('-') for w in items]
    cnt = Counter(num_sylls)
    exclude = set([k for k, v in cnt.most_common() if v < 3])
    items = [w for w in items if w.count('-') not in exclude]
    num_sylls = [w.count('-') for w in items]

    print(f'-> loaded {len(items)} items in total')

    # we stratify based on the number of syllables:
    train, rest, _, rest_sylls = split(items, num_sylls,
                        train_size=args.train_prop,
                        shuffle=True,
                        random_state=args.seed,
                        stratify=num_sylls)
    dev, test = split(rest,
                      train_size=0.5,
                      shuffle=True,
                      random_state=args.seed,
                      stratify=rest_sylls)

    print(f'# train items: {len(train)}')
    print(f'# dev test: {len(dev)}')
    print(f'# test items: {len(test)}')

    for items in ('train', 'dev', 'test'):
        with open(os.sep.join((args.split_dir, items + '.txt')), 'w') as f:
            f.write('\n'.join(eval(items)))

if __name__ == '__main__':
    main()