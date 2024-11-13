

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
        
        obeservations = sequence.outputseq
        num_states = len(self.transitions)
        num_observations = len(obeservations)
        
        Matrix = numpy.zeros((num_states, num_observations + 1))
        Matrix[0][0] = 1 # initial state
        
        for i in range(1, num_observations + 1):
            observation = obeservations[i-1]
            states = list(self.transitions.keys()) 
            for state in states:
                if state == '#':
                    continue
                sum = 0
                for state2 in states:  
                    s_idx = states.index(state)
                    sum += Matrix[states.index(state2)][i - 1] * self.transitions[state2][state] * self.emissions[state][observation]
                Matrix[s_idx][i] = round(sum,4)
        highest_prob = numpy.argmax(Matrix[:, num_observations])
        most_probable_state = states[highest_prob]
        print(Matrix)
        return most_probable_state 



    def viterbi(self, sequence):
        ## you do this. Given a sequence with a list of emissions, fill in the most likely
        ## hidden states using the Viterbi algorithm.
        obeservations = sequence.outputseq
        num_states = len(self.transitions)
        num_observations = len(obeservations)
        
        Matrix = numpy.zeros((num_states, num_observations + 1))
        Matrix[0][0] = 1
        V_matrix = numpy.zeros((num_states, num_observations + 1))
        T_matrix = numpy.zeros((num_states, num_observations + 1))
            
        for i in range(1, num_observations + 1):
            observation = obeservations[i-1]
            states = list(self.transitions.keys()) 
            for state in states:
                if state == '#':
                    continue
                largest_idx = 0
                curr_largest = 0
                sum = 0
                for v_index, state2 in enumerate(states): 
                    curr_prob = Matrix[states.index(state2)][i - 1] * self.transitions[state2][state] * self.emissions[state][observation]
                    s_idx = states.index(state)
                    sum += curr_prob
                    if curr_prob > curr_largest:
                        largest_idx = v_index
                        curr_largest = curr_prob
                Matrix[s_idx][i] = round(sum,4)
                V_matrix[s_idx][i] = largest_idx
                T_matrix[s_idx][i] = curr_largest
        
        print(Matrix)
        print(V_matrix)
        print(T_matrix)
        
        #print the most likely sequence
        most_likely = []
        highest_prob = numpy.argmax(Matrix[:, num_observations])
        most_likely.append(states[highest_prob])
        for i in range(num_observations, 0, -1):
            most_likely.append(states[int(V_matrix[highest_prob][i])])
        most_likely.reverse()
        print(most_likely)
        



if __name__ == "__main__":
    hmm = HMM()
    hmm.load('cat')
    # print(hmm.transitions)
    # print(hmm.emissions)
    # print(hmm.generate(10))
    # print(hmm.forward(Sequence(['#', 'happy', 'grumpy', 'hungry'], ['purr', 'silent', 'silent', 'meow', 'meow'])))
    
    print(hmm.viterbi(Sequence(['#', 'happy', 'grumpy', 'hungry'], ['purr', 'silent', 'silent', 'meow', 'meow'])))
    # hmm.viterbi(Sequence(['happy', 'grumpy', 'hungry'], ['meow', 'purr', 'meow']))
    


    # parser = argparse.ArgumentParser(description='HMM')
    # parser.add_argument('basename', type=str, help='basename for model files')