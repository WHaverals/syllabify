import argparse
import os

from sklearn.metrics import accuracy_score

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

from syllabification.Syllabify import Syllabifier
from syllabification.vectorization import SequenceVectorizer
from syllabification.modelling import build_model
import syllabification.utils as u

def main():
    parser = argparse.ArgumentParser(description='Splits available data in train-dev-test')
    parser.add_argument('--input_dir', type=str,
                        default='data/splits',
                        help='location of the splits folder')
    parser.add_argument('--model_dir', type=str,
                        default='model_s',
                        help='location of the model folder')
    parser.add_argument('--num_epochs', type=int,
                        default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float,
                        default=0.25,
                        help='Recurrent dropout')
    parser.add_argument('--num_layers', type=int,
                        default=2,
                        help='Number of recurrent layers')
    parser.add_argument('--retrain', default=False, action='store_true',
                        help='Retrain a model from scratch')
    parser.add_argument('--recurrent_dim', type=int,
                        default=30,
                        help='Number of recurrent dims')
    parser.add_argument('--emb_dim', type=int,
                        default=64,
                        help='Number of character embedding dims')
    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help='Batch size')
    parser.add_argument('--seed', type=int,
                        default=43432,
                        help='Random seed')
    args = parser.parse_args()
    print(args)
    
    train, dev, test = u.load_splits(args.input_dir, max_from_file=4000)

    train_words, train_Y = train
    dev_words, dev_Y = dev
    test_words, test_Y = dev

    v = SequenceVectorizer().fit(train_words)
    v_path = os.sep.join((args.model_dir, 'vectorizer.json'))
    v.dump(v_path)

    train_X = v.transform(train_words)
    dev_X = v.transform(dev_words)
    test_X = v.transform(test_words)

    train_Y = v.normalize_len(train_Y)
    dev_Y = v.normalize_len(dev_Y)
    test_Y = v.normalize_len(test_Y)

    train_Y = to_categorical(train_Y, num_classes=2)
    dev_Y = to_categorical(dev_Y, num_classes=2)
    test_Y = to_categorical(test_Y, num_classes=2)

    model = build_model(vectorizer=v, embed_dim=args.emb_dim,
                    num_layers=args.num_layers, lr=args.lr,
                    recurrent_dim=args.recurrent_dim,
                    dropout=args.dropout)

    model.summary()

    m_path = os.sep.join((args.model_dir, 'syllab.model'))

    if args.retrain:
        checkpoint = ModelCheckpoint(m_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)
            
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                    patience=1, min_lr=0.0001,
                                    verbose=1, min_delta=0.001)

        try:
            model.fit(train_X, train_Y, validation_data=[dev_X, dev_Y],
                    epochs=args.num_epochs, batch_size=args.batch_size,
                    shuffle=True, callbacks=[checkpoint, reduce_lr])
        except KeyboardInterrupt:
            print('\n' + '-' * 64 + '\n')
            pass
    
    model = load_model(m_path)
    
    # evaluate on test:
    test_silver = u.pred_to_classes(model.predict(test_X))
    test_gold = u.pred_to_classes(test_Y)

    gold_syll, pred_syll = [], []

    for test_item, gold, silver in zip(test_X, test_gold, test_silver):
        end = list(test_item).index(v.syll2idx['<EOS>'])
        gold_syll.append(tuple(gold[1:end]))
        pred_syll.append(tuple(silver[1:end]))

    test_acc_syll = accuracy_score([i for s in gold_syll for i in s],
                                   [i for s in pred_syll for i in s])
    test_acc_token = accuracy_score([str(s) for s in gold_syll], 
                                    [str(s) for s in pred_syll])
    print('test acc (char):', test_acc_syll)
    print('test acc (token):', test_acc_token)

    with open(os.sep.join((args.model_dir, 'silver.txt')), 'w') as f:
        for token, pred in zip(test_words, test_silver):
            f.write(u.stringify(token, pred) + '\n')
    
    """
    # run Bouma et al:
    utils.run_bouma_et_al()

    # run the LSTM:
    s.syllabify(inp='../data/test_input.txt',
                outp='../data/lstm_output.txt')

    # evaluate both approaches:
    print('-> lstm scores:')
    s.evaluate(goldp='../data/test_gold.txt',
               silverp='../data/lstm_output.txt')

    print('-> Bouma et al scores:')
    s.evaluate(goldp='../data/test_gold.txt',
               silverp='../data/bouma_et_al_output.txt')
    """

if __name__ == '__main__':
    main()