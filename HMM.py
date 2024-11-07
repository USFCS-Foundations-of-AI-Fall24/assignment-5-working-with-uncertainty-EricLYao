

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
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.



if __name__ == "__main__":
    hmm = HMM()
    hmm.load('cat')
    # print(hmm.transitions)
    # print(hmm.emissions)
    print(hmm.generate(10))
    # hmm.forward(Sequence(['happy', 'grumpy', 'hungry'], ['meow', 'purr', 'meow']))
    # hmm.viterbi(Sequence(['happy', 'grumpy', 'hungry'], ['meow', 'purr', 'meow']))
    # hmm.viterbi(Sequence(['happy', 'grumpy', 'hungry'], ['meow', 'purr', 'meow']))


    # parser = argparse.ArgumentParser(description='HMM')
    # parser.add_argument('basename', type=str, help='basename for model files')