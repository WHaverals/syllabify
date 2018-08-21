from Syllabify import Syllabifier

def main():

    syllabifier = Syllabifier(model_dir = 'model_s', load = True)
    sample_words = []

    with open('../data/sample_words_cdrom.txt', 'r') as f:
        for word in f:
            sample_words.append(word)

    syllabified_sample_words = syllabifier.syllabify(data=sample_words, outp=None)

    with open('../data' + '/' + 'syllabified_sample_words.txt', 'w') as f:
        for word in syllabified_sample_words:
            f.write(''.join(word))
    
if __name__ == '__main__':
    main()