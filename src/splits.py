import argparse
import shutil
import os

from sklearn.model_selection import train_test_split as split

def main():
    parser = argparse.ArgumentParser(description='Splits available data in train-dev-test')
    parser.add_argument('--input_dir', type=str,
                        default='../data/syllabified_crm.txt',
                        help='location of the full data file')
    parser.add_argument('--split_dir', type=str,
                        default='../data/splits',
                        help='location of the train-dev-test files')
    parser.add_argument('--train_prop', type=float,
                        default=.8,
                        help='Proportion of training items (dev and test are equal-size)')
    parser.add_argument('--seed', type=int,
                        default=43432,
                        help='Proportion of training items (dev and test are equal-size)')
    args = parser.parse_args()
    print(args)

    try:
        shutil.rmtree(args.split_dir)
    except FileNotFoundError:
        pass
    os.mkdir(args.split_dir)

    with open(args.input_dir, 'r') as f:
        items = [l.strip() for l in f if l.strip()]

    print(f'-> loaded {len(items)} items in total')

    train, rest = split(items,
                        train_size=args.train_prop,
                        shuffle=True,
                        random_state=args.seed)
    dev, test = split(rest,
                      train_size=0.5,
                      shuffle=True,
                      random_state=args.seed)

    print(f'# train items: {len(train)}')
    print(f'# dev test: {len(dev)}')
    print(f'# test items: {len(test)}')

    for items in ('train', 'dev', 'test'):
        with open(os.sep.join((args.split_dir, items + '.txt')), 'w') as f:
            f.write('\n'.join(eval(items)))

if __name__ == '__main__':
    main()