# In order to know how well the syllabifier is performing (on different texts, from different periods),
# we take a random sample of unique words from the Corpus of Middle Dutch rhymed texts (cm)

# 1. parse xml to txt's
# 2. get rndom sample (n=1000) from these txts
# 3. apply automatic syllabifier onto sample
# 4. manually check the model's performance

import re
import os
import html
import shutil

from random import sample
from unidecode import unidecode
from bs4 import BeautifulSoup

from Syllabify import Syllabifier

def main():

    syllabifier = Syllabifier(model_dir = '../models', load = True)
   
    LACUNA = re.compile(r'\.\.+')

    output_dir = '../data/orig_txt'

    try:
        shutil.rmtree(output_dir)
    except:
        pass
    
    os.mkdir(output_dir)

    # cleaning xml

    all_lines = []
    for entry in os.scandir('../data/corpus_mnl_rijm'):
        if not entry.path.endswith('.xml'):
            continue
        print('--> parsing:', entry.path)

        with open(entry.path, 'r') as f:
            xml_text = f.read()
            
            xml_text = xml_text.replace('&oudpond;', '')
            xml_text = xml_text.replace('&supm;', 'm')
            xml_text = xml_text.replace('&supM;', 'm')
            xml_text = xml_text.replace('&supc;', 'c')
            xml_text = xml_text.replace('&supt;', 't')
            xml_text = xml_text.replace('&supn;', 'n')
            xml_text = xml_text.replace('&sups;', 's')
            xml_text = xml_text.replace('&supd;', 'd')
            xml_text = xml_text.replace('&supc;', 'c')
            xml_text = xml_text.replace('&uring;', 'u')
            xml_text = xml_text.replace('&lt;', '')
            xml_text = xml_text.replace('&gt;', '')
            xml_text = html.unescape(xml_text)

        soup = BeautifulSoup(xml_text, 'html.parser')

        lines = []
        for line in soup.find_all('l'):
            if line.has_attr('parse'):
                continue
            text = line.get_text().strip()
            if (not text) or (re.search(LACUNA, text)):
                continue
            else:
                lines.append(text)
        
        clean_lines = []
        for line in lines:
            line = line.lower()
            clean_line = ''
            for char in line:
                if char.isalpha() or char.isspace():
                    clean_line += char
            clean_line = clean_line.strip()
            if clean_line:
                clean_lines.append(clean_line)
                all_lines.append(clean_line)
        
        with open(output_dir + '/' + entry.name.replace('.xml', '.txt'), 'w') as f:
            for line in clean_lines:
                f.write(''.join(line) + '\n')
    
    with open(output_dir + '/' + 'all_lines.txt', 'w') as fw:
        for line in all_lines:
            fw.write(''.join(line) + '\n')
    
    words = []

    for line in all_lines:
        for word in line.split():
            words.append(word)

    unique_words = set(words)
    sample_words = sample(unique_words, 2000)
    
    with open('../data' + '/' + 'sample_words.txt', 'w') as fw:
        for word in sample_words:
            fw.write(''.join(word) + '\n')

    syllabified_sample_words = syllabifier.syllabify(data=sample_words, outp=None)
    
    with open('../data' + '/' + 'syllabified_sample_words.txt', 'w') as fw:
        for word in syllabified_sample_words:
            fw.write(''.join(word) + '\n')
    
    print(sample_words, syllabified_sample_words)
    print('Unique words in Rhymed text corpus Middelnederlands:', len(unique_words))
    # Unique words in Rhymed text corpus Middelnederlands: 108196

if __name__ == '__main__':
    main()