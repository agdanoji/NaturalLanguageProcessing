#!/bin/python
from nltk.corpus import wordnet
from word2number import w2n
import string
def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """

    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    #if word.isdigit():
        #ftrs.append("IS_DIGIT")

    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # to check if string of letters in word or the word itself is a digit
    try:
        if word.isdigit() or w2n.word_to_num(word).isdigit():
            ftrs.append("IS_DIGIT")
    except: ''

    #to check word validity
    if wordnet.synsets(word):
        ftrs.append("IS_VALID")

    # checks verb endings in word
    v=['ed','ing','s','d','ies','es']
    if word[-2:] in v:
        ftrs.append("HAS_VCONJ")

    #checks adjective endings in word
    if word[-2:] is "er" or word[0:2] is "un":
        ftrs.append("HAS_ADJCONJ")

    #checks if words has -
    #if '-' in word:
        #ftrs.append("HAS_'-'")

    #checks adverb endings in word
    adv=['ly','ful']
    if word[-2:] in adv:
        ftrs.append("HAS_AVCONJ")

    #checks if words start with capital letter
    if word[0].isupper():
        ftrs.append("CAP_START")
    #Checks if word contains @ or #
    if "#" in word or "@" in word :
        ftrs.append("HAS_@#")
    # checks if word is conjunction
    #conj=['and','if','as','but','or','while','as']
    #if word in conj:
        #ftrs.append("IS_CONJ")

    # checks if word contains following
    #dot=['.',',','?','...','!',':',';','!!']
    #if word in dot:
        #ftrs.append("IS_DOT")

    # checks if word is determinant
    #det=['the','a','an']
    #if word in det:
        #ftrs.append("IS_DET")

    # checks if word is a pronoun
    #pn=['I','i','me','he','she','her','you','u','it','that','they','each','few','many','who','whoever','whose','someone']
    #if word in pn:
        #ftrs.append("IS_PN")

    # checks if word contains '
    if "'" in word:
        ftrs.append("HAS_'")

    # checks if word is plural
    if word[-1:] is "s":
        ftrs.append("IS_PLURAL")

    # checks if word is punctuation
    if word in string.punctuation:
        ftrs.append("IS_PUNCT")

    # to check if sentence is long enough
    #if len(sent)>10:
        #ftrs.append("IS_LONG")



    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        #if i >1:
            #for pf in token2features(sent, i-2, add_neighs = False):
                #ftrs.append("PPREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)
        #if i < len(sent) - 2:
            #for pf in token2features(sent, i + 2, add_neighs=False):
                #ftrs.append("NNEXT_" + pf)


    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
