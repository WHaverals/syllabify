import os

from syllabification.vectorization import SequenceVectorizer
from syllabification import utils as u

from keras.models import load_model


def main():
    """
    Command line demo to explore the syllabification model.
    """
    model_dir = 'model_s'
    v_path = os.sep.join((model_dir, 'vectorizer.json'))
    v = SequenceVectorizer.load(v_path)
    m_path = os.sep.join((model_dir, 'syllab.model'))
    model = load_model(m_path)

    info = '\n' * 10
    info += "######################################################\n"
    info += "##### Syllabification Demo for Middle Dutch #########\n"
    info += "#####################################################\n\n"
    print(info)

    phrase = 'Enter a Middle Dutch word (or type QUIT to stop): '
    word = ''
    
    while True:
        word = input(phrase).strip()
        if word == 'QUIT':
                break
        X = v.transform([word])
        pred = u.pred_to_classes(model.predict(X))[0]
        segmented = u.stringify(word, pred)
        print('Segmentation proposed: '+ segmented)

if __name__ == '__main__':
    main()