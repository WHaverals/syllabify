import argparse

import syllabification.utils as u

def vectorize(words_, labels_, window):
    X, Y = [], []

    for word, labels in zip(words_, labels_):
        word = ['<BOS>'] + word + ['<EOS>']
        labels = ['<BOS>'] + labels + ['<EOS>']

        for idx, (char, lab) in enumerate(zip(word, labels)):
            # skip BOS/EOS
            if idx in (0, len(word) - 1):
                continue
            
            left_context = [word[idx-(t+1)] for t in range(window) if idx-(t+1) >= 0][::-1]
            while len(left_context) < window:
                    left_context = ['<PAD>'] + left_context

            right_context = [word[idx+t+1] for t in range(window) if idx+t+1 < len(word)]
            while len(right_context) < window:
                    right_context += ['<PAD>']
            
            features = left_context + [char] + right_context
            features = [str(i + 1)+'-' + f for i, f in enumerate(features)]
            
            X.append(features)
            Y.append(labels[idx])
    
    return X, Y


def main():
    parser = argparse.ArgumentParser(description='Trains baseline model')
    parser.add_argument('--input_dir', type=str,
                        default='data/splits',
                        help='location of the splits folder')
    parser.add_argument('--model_dir', type=str,
                        default='model_b',
                        help='location of the model folder')
    parser.add_argument('--window', type=int,
                        default=3,
                        help='Length of window around focus character (symmetric)')
    args = parser.parse_args()
    print(args)

    train, dev, test = u.load_splits(args.input_dir)

    train_words, train_Y = train
    dev_words, dev_Y = dev
    test_words, test_Y = test

    train_X, train_Y = vectorize(train_words, train_Y, window=args.window)
    dev_X, dev_Y = vectorize(dev_words, dev_Y, window=args.window)
    test_X, test_Y = vectorize(test_words, test_Y, window=args.window)


if __name__ == '__main__':
    main()