import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """

            # Initialize placeholders
            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.float32, shape=[Config.batch_size, parsing_system.numTransitions()])
            self.test_inputs = tf.placeholder(tf.int32)

            # Initilaze arguments for forward pass
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            embed = tf.reshape(embed, [Config.batch_size, -1])

            # initialise weights and bias with random normal
            weights_input = tf.Variable(tf.random_normal([Config.n_Tokens * Config.embedding_size, Config.hidden_size1],0.0,stddev=0.1))
            '''
            Code for implementation of different experiments
                             #'h2': tf.Variable(tf.random_normal([18 * Config.embedding_size, Config.hidden_size1], 0.0,stddev=0.1)),
                             #'h3': tf.Variable(tf.random_normal([12* Config.embedding_size, Config.hidden_size1], 0.0, stddev=0.1))}

                             #'h2':tf.Variable(tf.random_normal([Config.hidden_size1,Config.hidden_size2],0.0,stddev=0.1)),
                             #'h3':tf.Variable(tf.random_normal([Config.hidden_size2,Config.hidden_size3],0.0,stddev=0.1))}
                             
             # tf.Variable(tf.random_poisson([0.5]*Config.hidden_size1,[Config.n_Tokens*Config.embedding_size]))
            '''


            biases_input =tf.Variable(tf.zeros([Config.hidden_size1]))
                           #'b2':tf.Variable(tf.zeros([Config.hidden_size2])),
                           #$'b3': tf.Variable(tf.zeros([Config.hidden_size3]))}

            weights_output =tf.Variable(tf.random_normal([Config.hidden_size3, parsing_system.numTransitions()],0.0,stddev=0.1))

            #tf.Variable(tf.random_poisson([0.5]*parsing_system.numTransitions(),[Config.hidden_size3]))


            # Get predictions
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)



            # Pick the transition with highest score
            truelable = tf.argmax(self.train_labels, axis=1)
            # l2-regularization
            theta = tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_input)+ tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(self.embeddings)
            # Loss function implementation
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=truelable, logits=self.prediction)) + (Config.lam * theta)

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)

            # Gradient clipping implementation
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)


            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_inpu, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        # For implementation of passing different embeddings at different hidden layers
        #e1 = embed[:, 0:900]
        #e2 = embed[:, 900:1800]
        #e3 = embed[:, 1800:2400]

        # using cubic activation function
        a1 = tf.matmul(embed, weights_input)
        l1 = tf.pow(tf.add(a1, biases_inpu),3)

        """
        # using 3 hidden layers
        #a2 =tf.pow(tf.matmul(e2, weights_input['h2']),3)
        #l2 = tf.nn.relu(tf.add(a2, biases_inpu['b2']))
        #a3 = tf.pow(tf.matmul(e3, weights_input['h3']),3)
        #l3 = tf.nn.relu(tf.add(a3, biases_inpu['b3']))
        
        # using different activation functions 
        # sigmoid
        #h=tf.sigmoid(tf.add(a, biases_inpu))
        # tanh
        #h=tf.tanh(tf.add(a1, biases_inpu))
        # relu
        #h=tf.nn.relu(tf.add(a, biases_inpu))
        #h=tf.add(l1,biases_inpu)
        """

        p = tf.matmul(l1, weights_output)
        return p



def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    s_w = []  # for words n_w=18
    s_t = []  # for POS tags n_t=18
    s_l = []  # for lables n_l=12

    # top 3 words of the stack
    for i in range(0, 3):
        s_w.append(getWordID(c.getWord(c.getStack(i))))

    # the POS stags of top 3 words of the stack
    for i in range(0, 3):
        s_t.append(getPosID(c.getPOS(c.getStack(i))))

    # top 3 words of the buffer
    for i in range(0, 3):
        s_w.append(getWordID(c.getWord(c.getBuffer(i))))

    # the POS stags of top 3 words of the buffer
    for i in range(0, 3):
        s_t.append(getPosID(c.getPOS(c.getBuffer(i))))

    # get the word, pos and lable for the first and second leftmost children of the top two words on the stack
    for i in range(0, 2):
        k = c.getStack(i)
        flc = c.getLeftChild(k, 1)
        slc = c.getLeftChild(k, 2)
        s_w.append(getWordID(c.getWord(flc)))
        s_w.append(getWordID(c.getWord(slc)))
        s_t.append(getPosID(c.getPOS(flc)))
        s_t.append(getPosID(c.getPOS(slc)))
        s_l.append(getLabelID(c.getLabel(flc)))
        s_l.append(getLabelID(c.getLabel(slc)))

    # get the word, pos and lable for the first and second rightmost children of the top two words on the stack
    for i in range(0, 2):
        k = c.getStack(i)
        frc = c.getRightChild(k, 1)
        src = c.getRightChild(k, 2)
        s_w.append(getWordID(c.getWord(frc)))
        s_w.append(getWordID(c.getWord(src)))
        s_t.append(getPosID(c.getPOS(frc)))
        s_t.append(getPosID(c.getPOS(src)))
        s_l.append(getLabelID(c.getLabel(frc)))
        s_l.append(getLabelID(c.getLabel(src)))

    # get the word, pos and lable for the leftmost of leftmost children of the top two words on the stack
    for i in range(0, 2):
        k = c.getStack(i)
        flc = c.getLeftChild(k, 1)
        fllc = c.getLeftChild(flc, 1)
        s_w.append(getWordID(c.getWord(fllc)))
        s_t.append(getPosID(c.getPOS(fllc)))
        s_l.append(getLabelID(c.getLabel(fllc)))

    # get the word, pos and lable for the rightmost of rightmost children of the top two words on the stack
    for i in range(0, 2):
        k = c.getStack(i)
        frc = c.getRightChild(k, 1)
        frrc = c.getRightChild(frc, 1)
        s_w.append(getWordID(c.getWord(frrc)))
        s_t.append(getPosID(c.getPOS(frrc)))
        s_l.append(getLabelID(c.getLabel(frrc)))

    features = []
    features.extend(s_w)
    features.extend(s_t)
    features.extend(s_l)
    return features



def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    #len(sents)
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

