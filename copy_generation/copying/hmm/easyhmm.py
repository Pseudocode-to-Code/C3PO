import pandas as pd
import numpy as np

class EasyHMM:

    def __init__(self, tagged_pairs_corpus: pd.Series) -> None:
        """
        Parameters:
        """

        self.A = None
        self.B = None
        self.pi = None

        self.tagged_pairs_corpus = tagged_pairs_corpus
        self.total_corpus = None


    def _compute_B(self, word: str, tag: int):
        """
        Compute emission probability

        Parameters:

        word: pseudocode word to check for
        tag: 0 or 1
        """

        # Pairs with the tag specified
        tag_list = [pair for pair in self.total_corpus if pair[1]==tag]

        # Number of times that tag occurred
        count_tag = len(tag_list)

        # Pairs with the word specified
        w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]

        # Total number of times the passed word occurred as the passed tag
        count_w_given_tag = len(w_given_tag_list)
    
        return (count_w_given_tag, count_tag)
    
    def _compute_A(self, tag1, tag2):
        """
        Compute transition probability

        Parameters:
        tag1: tag at time t
        tag2: tag at time t+1
        """

        count_t1 = 0
        count_t2_t1 = 0

        for row in self.tagged_pairs_corpus:

            # List of sequence of tags (hidden states)
            tags = [pair[1] for pair in row]

            # Number of t1 occurrences
            count_t1 += len([t for t in tags if t==tag1])
            
            # Count of t1 followed by t2 occurrences
            for index in range(len(tags)-1):
                if tags[index] == tag1 and tags[index+1] == tag2:
                    count_t2_t1 += 1
        
        return (count_t2_t1, count_t1)

    def fit(self):
        """
        Fit an easy HMM based on output_sequences and hidden_sequences data
        """

        self.total_corpus = [tup for row in self.tagged_pairs_corpus for tup in row]
        self.tags = set([pair[1] for pair in self.total_corpus])

        self.A = np.zeros(len(self.tags))

        for i, t1 in enumerate(list(self.tags)):
            for j, t2 in enumerate(list(self.tags)): 
                count_t2_t1, count_t1 = self._compute_A(t1, t2)
