==========================================================================================

         Introduction to Natural Language Processing Assignment 3
		 
Student ID: 111493647
==========================================================================================

1)apply() in ParsingSystem.py:
Implemented arc standard algorithm based on the paper Incrementally in Deterministic Dependency Parsing(Nirve, 2004).
The configuration of arc standard algorithm is:(s,b,A)
Here s=stack , b=buffer, A= set of dependency Arcs.
If we have a sentence as (w1 w2 w3 w4 ... wn) then:
Initial Configuration is s=[ROOT] , b=[w1,w2,w3 ,w4,...,wn] and A=[].
Final Configuration is s=[ROOT], b=[],A=[final set of dependency arcs].
s1-sn --> elements in stack
b1-bn --> elements in buffer 
Three operations are as follows:
s1= topmost element in stack, s2=second element from top in the stack.
-Left-reduce(l)- if |s|>2 , add arc s1->s2 to A with label l and remove s2 from the stack.
-Right-reduce(l)- if |s|>2 , add arc s2->s1 to A with label l and remove s1 from the stack.
-Shift- move a word from the front of buffer to the top of the stack if buffer is not empty.
Return the set A with changes made.


2)getFeatures() in DependencyParser.py:
The feature vectors contains the index's or Id's of the below specified features.
The number of features used in total are 48 as follow:

	->3-top 3 words of the stack
	->3-the POS stags of top 3 words of the stack
	->3-top 3 words of the buffer
	->3-the POS stags of top 3 words of the buffer
	->12-the word, pos and label for the first and second leftmost children of the top two words on the stack
	->12-the word, pos and label for the first and second rightmost children of the top two words on the stack
	->6-the word, pos and label for the leftmost of leftmost children of the top two words on the stack
	->6-the word, pos and label for the rightmost of rightmost children of the top two words on the stack
	
3)forward_pass() in DependencyParser.py:
Build a basic neural network with a single hidden layer using cubic activation function.We pass the feature vector as specified above to the input layer, apply activation function to map it to the hidden layer after adding bias. Softmax is applied at the end to take care of multi-class probabilities as suggested by the research paper.  

4)Dependency Parser using Neural Network Implementation:
	a)Initializations:
	- train_input: tensor of size batch_size * number of tokens 
	- test-label: tensor of size batch_size * number of transitions
	- test_input: initialize tensor 
	-embed-: size [batch_size, -1]
	-weight-input: using random normal initialization of size (n_Tokens * embedding_size, hidden layer size)
	-weight-output: using random normal initialization of hidden layer size* number of transitions
	-biases- initialize to zero 
	
	b)Loss function Implementation:
L(θ)=−\summ(log pt)+0.5*lam*theta
Finally we need to implement loss function.The goal here is to minimize the cross-entropy loss. Theta is l2-regularization for the set of all parameters ie weights , biases and embedding. The word embedding used are pre-trained with embedding size of 50. Greedy decoding is performed in parsing in which we pick the transition with the highest score, by taking argmax. Used sparse_softmax_cross_entropy_with_logits() function of tensor-flow to compute loss.

