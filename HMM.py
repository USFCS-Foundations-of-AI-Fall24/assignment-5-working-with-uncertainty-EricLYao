

import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    def load(self, basename):
        """loads a model from files basename.trans and basename.emit"""
        trans_file = basename + ".trans"
        emit_file = basename + ".emit"
        
        ## .trans file parsing
        with open(trans_file) as f :
            for line in f:
                line = line.strip().split()
                if len(line) == 2:
                    self.transitions[line[0]] = line[1]
                else:
                    if line[0] not in self.transitions:
                        self.transitions[line[0]] = {}
                    self.transitions[line[0]][line[1]] = float(line[2])
        
        ## .emit file parsing
        with open(emit_file) as f:
            for line in f:
                line = line.strip().split()
                if len(line) == 2:
                    self.emissions[line[0]] = line[1]
                else:
                    if line[0] not in self.emissions:
                        self.emissions[line[0]] = {}
                    self.emissions[line[0]][line[1]] = float(line[2])

        

   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        stateseq = []
        outputseq = []
        for i in range(n):
            if i == 0:
                state = numpy.random.choice(list(self.transitions["#"].keys()), p=list(self.transitions["#"].values()))
            else:
                state = numpy.random.choice(list(self.transitions[stateseq[i-1]].keys()), p=list(self.transitions[stateseq[i-1]].values()))
            output = numpy.random.choice(list(self.emissions[state].keys()), p=list(self.emissions[state].values()))
            stateseq.append(state)
            outputseq.append(output)
        return Sequence(stateseq, outputseq)

    def forward(self, sequence):
        ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
        ## determine the most likely sequence of states.
        observations = sequence.outputseq
        states = list(self.transitions.keys()) 
        Matrix = numpy.zeros((len(states), len(observations) + 1))
        Matrix[0][0] = 1
        
        for i in range(1, len(observations) + 1):
            observation = observations[i-1]
            for s_index, state in enumerate(states):
                if state != '#':
                    sum = 0
                    for s2_index, state2 in enumerate(states):  
                        if state in self.transitions[state2] and observation in self.emissions[state]:
                            sum += Matrix[s2_index][i - 1] * self.transitions[state2][state] * self.emissions[state][observation]
                    Matrix[s_index][i] = sum
        highest_prob = numpy.argmax(Matrix[:, len(observations)])
        most_probable_state = states[highest_prob]
        return most_probable_state 



    def viterbi(self, sequence):
        ## you do this. Given a sequence with a list of emissions, fill in the most likely
        ## hidden states using the Viterbi algorithm.
        observations = sequence.outputseq
        states = list(self.transitions.keys()) 
        
        V_matrix = numpy.zeros((len(states), len(observations) + 1))
        V2_matrix = numpy.zeros((len(states), len(observations) + 1))
        V_matrix[0][0] = 1
            
        for i in range(1, len(observations) + 1):
            observation = observations[i-1]
            for s_index, state in enumerate(states):
                if state != '#':
                    largest_idx = 0
                    curr_largest = 0
                    for v_index, state2 in enumerate(states): 
                        if state in self.transitions[state2] and observation in self.emissions[state]:
                            curr_prob = V_matrix[v_index][i - 1] * self.transitions[state2][state] * self.emissions[state][observation]
                            if curr_prob > curr_largest:
                                largest_idx = v_index
                                curr_largest = curr_prob
                    V_matrix[s_index][i] = curr_largest
                    V2_matrix[s_index][i] = largest_idx

        most_likely_seq = []
        most_likely_seq.append(int(numpy.argmax(V_matrix[:, len(observations)])))
        for i in range(len(observations), 0, -1):
            most_likely_seq.append(int(V2_matrix[int(most_likely_seq[-1])][i]))
        most_likely_seq.reverse()
        most_likely_states = []
        for i in most_likely_seq:
            most_likely_states.append(states[i])
            
        return most_likely_states


if __name__ == "__main__":
    # hmm = HMM()
    # hmm.load('lander')
    # testSeq = hmm.generate(10)
    # print(' '.join(testSeq.outputseq))
    
    # print(hmm.forward(Sequence(['#', 'happy', 'grumpy', 'hungry'], ['purr', 'purr', 'meow', 'silent', 'meow', 'purr', 'meow', 'meow', 'silent', 'purr'])))
    
    # print(hmm.viterbi(Sequence(['#', 'happy', 'grumpy', 'hungry'], ['purr', 'silent', 'silent', 'meow', 'meow'])))
    # print("Most likely hidden states: ", hmm.viterbi(testSeq))
    
    parser = argparse.ArgumentParser(description='Hidden Markov Model')
    parser.add_argument('basename', type=str, help='basename without file extension')
    parser.add_argument('obsfile', type=str, help='observation file')
    parser.add_argument('--forward', action='store_true', help='run forward algorithm')
    parser.add_argument('--viterbi', action='store_true', help='run Viterbi algorithm')
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)

    with open(args.obsfile) as f:
        for line in f:
            line = line.strip().split()
            if line: 
                seq = Sequence([hmm.transitions.keys], line)
                if args.viterbi:
                    print("Most likely hidden states: ", hmm.viterbi(seq))
                elif args.forward:
                    print("Most likely final state: ", hmm.forward(seq))
                else:
                    print("Not a valid function")