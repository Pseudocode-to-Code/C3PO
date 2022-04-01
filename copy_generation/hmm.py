from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np


class HMM(torch.nn.Module):
    """
    HMM with discrete observations
    """

    @classmethod
    def _get_hid_obs_states_lists(cls, train_sequences: List[List[Tuple[str, str]]]) -> Tuple[List]:
        """
        Get hidden and observation states lists
        """
        hidden_states = set()
        observation_states = set()
        
        for row in train_sequences:
            for word, tag in row:
                hidden_states.add(tag)
                observation_states.add(word)

        return list(hidden_states), list(observation_states)

    def __init__(self, num_obs: int, num_hid: int, train_sequences: List[List[Tuple[str, str]]]):
        """
        Parameters:

        num_obs: number of possible observations
        num_hidden: number of possible hidden states
        """
        super(HMM, self).__init__()
        self.num_obs = num_obs
        self.num_hid = num_hid

        # A
        self.A = torch.nn.Parameter(F.softmax(torch.randn(self.num_hid, self.num_hid), dim=0))

        # B
        self.B = torch.nn.Parameter(F.softmax(torch.randn(self.num_hid, self.num_obs), dim=0))

        # pi (random normal initialzation)
        self.pi = torch.nn.Parameter(F.softmax(torch.randn(self.num_hid), dim=0))

        # Lists of hidden and observation states
        self.hidden_states, self.observation_states = HMM._get_hid_obs_states_lists(train_sequences)

        # Map of observation states to indices
        self.observation_states_map = {k: v for v, k in enumerate(self.observation_states)}

        # use the GPU
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda: self.cuda()

    def hmm_forward(self, output_sequence: List[str]) -> torch.TensorType:
        """
        Forward algorithm

        Computes and returns alpha table

        Parameters:

        output_sequence: a single sequence of outputs -> eg ['set', 'l', 'to', '3']
        """

        # log domain
        alpha = torch.zeros(len(self.hidden_states), len(output_sequence))


        for t in range(len(output_sequence)):
            if t == 0:
                # First timestep, use pi                
                alpha[:,t] = torch.log(self.pi)
            else:
                alpha[:,t] = torch.logsumexp(torch.add(alpha[:,t-1:t], self.A), dim=0)
            
            current_output_state = self.observation_states_map[output_sequence[t]]
            alpha[:,t] += self.B[:,current_output_state]

        return alpha

    def hmm_backward(self, output_sequence: List[str]) -> torch.TensorType:
        """
        Backward algorithm

        Computes and returns beta table

        Parameters:

        output_sequence: a single sequence of outputs -> eg ['set', 'l', 'to', '3']
        """

        # log domain
        beta = torch.zeros(len(self.hidden_states), len(output_sequence))


        for t in range(len(output_sequence)-1, -1, -1):
            current_output_state = self.observation_states_map[output_sequence[t]]

            if t == len(output_sequence)-1:
                # First timestep, use pi                
                beta[:,t] = 1
            else:
                # torch.logsumexp(torch.add(self.B[:,current_output_state], self.A), dim=1)

                beta[:,t] = torch.logsumexp(torch.add(self.B[:,current_output_state], self.A), dim=1)

        return beta
        

    def baum_welch(self, train_sequences: List[List[Tuple[str, str]]], n_iter: int=100):
        """
        Run baum-welch algorithm to tune the transition_model and emission_model
        """


        print('-----Iter -1-----')
        print('pi')
        print(self.pi)
        print('A')
        print(self.A)
        print('B')
        print(self.B)
        
        for it in range(n_iter):

            K = len(train_sequences)

            A_hat_numerator = torch.zeros(self.A.shape)
            A_hat_denominator = torch.zeros(self.A.shape)
            B_hat_numerator = torch.zeros(self.B.shape)
            B_hat_denominator = torch.zeros(self.B.shape)

            pi_hat = torch.zeros(self.pi.shape)

            for k, cur_input in enumerate(train_sequences):

                # Expectation step - compute xi and gamma

                alpha = self.hmm_forward(cur_input)
                beta = self.hmm_backward(cur_input)

                T = len(cur_input)

                # calculate xi and gamma
                xi = torch.zeros(self.A.shape[0], self.A.shape[0], T)
                gamma = torch.zeros(self.A.shape[0], T)

                for t, word in enumerate(cur_input):
                    
                    for i in range(self.A.shape[0]):
                        denominator = torch.matmul(torch.exp(alpha[:,t]), torch.exp(beta[:,t]))

                        gamma[i,t] = torch.exp(alpha[i,t])*torch.exp(beta[i,t])
                        gamma[i,t] /= denominator

                        if t == T-1: continue

                        for j in range(self.A.shape[1]):
                            # From state i to j
                            xi[i,j,t] = torch.exp(alpha[i,t])*self.A[i,j]*self.B[j,self.observation_states_map[cur_input[t+1]]]*torch.exp(beta[j,t+1])
                            xi[i,j,t] /= denominator

                        

                # Maximization step - compute A and B
                
                pi_hat += gamma[:,0]

                for i in range(self.A.shape[0]):
                    B_hat_denominator[i,:] += torch.sum(gamma[i,:])

                    for t, word in enumerate(cur_input):
                        

                        B_hat_numerator[i,self.observation_states_map[word]] += gamma[i,t]

                        for j in range(self.A.shape[1]):
                            A_hat_numerator[i,j] += xi[i,j,t]
                            A_hat_denominator[i,j] += gamma[i,t]

            # k examples done
            self.pi = torch.nn.Parameter(pi_hat / K)
            self.A = torch.nn.Parameter(A_hat_numerator/A_hat_denominator)
            self.B = torch.nn.Parameter(B_hat_numerator/B_hat_denominator)

            print(f'-----Iter {it}-----')
            print('pi')
            print(self.pi)
            print('A')
            print(self.A)
            print('B')
            print(self.B)
    

    def train_for_viterbi(self, *, hidden_sequences, observation_sequences, hidden_state_set=None, observation_state_set=None):
        """
        Find values for A and B matrices
        """


        if hidden_state_set is not None:
            self.hidden_states = list(hidden_state_set)
        else:
            self.hidden_states = list(set([word for sent in hidden_sequences for word in sent]))

        assert len(self.hidden_states) == self.N

        if observation_state_set is not None:
            self.observation_states = list(observation_state_set)
        else:
            self.observation_states = list(set([word for sent in observation_sequences for word in sent]))

        assert len(self.observation_states) == self.M


if __name__ == '__main__':

    obs = [['apple', 'banana', 'carrot'], ['banana', 'carrot'], ['apple', 'banana'], ['carrot']]
    hid = [[1, 0, 1], [1, 0], [0, 0], [1]]

    train_seq = []

    for i in range(len(obs)):
        train_seq.append(list(zip(obs[i], hid[i])))

    hmm = HMM(num_obs=3, num_hid=2, train_sequences=train_seq)

    print(hmm.hmm_forward(['carrot', 'apple', 'apple']))
    print(hmm.hmm_backward(['carrot', 'apple', 'apple']))
    hmm.baum_welch(obs, 3)

    # hmm.(hidden_sequences=hid, observation_sequences=obs)

