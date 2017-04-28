import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.feature_count = len(self.X[0])

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        selected_model = None
        selected_model_bic_value = float("inf")

        # implement model selection based on BIC scores
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                parameters = n * (n - 1) + (n - 1) + 2 * self.feature_count * n
                logL = hmm_model.score(self.X, self.lengths)
                current_bic_value = -2 * logL + parameters * np.log(len(self.X))
                if current_bic_value < selected_model_bic_value:  # lower scores are better
                    selected_model = hmm_model
                    selected_model_bic_value = current_bic_value
            except:
                if self.verbose:
                    print("failure with word {} with {} states".format(self.this_word, n))

        return selected_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        selected_model = None
        selected_model_dic_value = 0

        # implement model selection based on DIC scores
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                logP = hmm_model.score(self.X, self.lengths)
                M = len(logP)
                sum_log_p = 0
                for word, (X, l) in self.hwords.items():
                    if word != self.this_word:
                        sum_log_p = sum_log_p + hmm_model.score(X, l)
                    current_dic_value =  logP - 1/(M - 1)*sum_log_p
                    if current_dic_value > selected_model_dic_value:  # higher scores are better
                            selected_model = hmm_model
                            selected_model_dic_value = current_dic_value
                return selected_model
            except Exception as e:
                return self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        selected_model = None
        selected_model_cv_value = float("-inf")
        n_splits = len(self.sequences)
        if n_splits <= 1:
            return None


        # implement model selection based on BIC scores
        for n in range(self.min_n_components, self.max_n_components + 1):
            kf = KFold(n_splits, shuffle=True, random_state=self.random_state)

            # Try out each fold; train on the training set and score against the test set
            for i_train, i_test in kf.split(self.sequences):
                try:
                    X_train, len_train = combine_sequences(i_train, self.sequences)
                    hmm_model = self.base_model(n, X_train, len_train)
                    X_test, len_test = combine_sequences(i_test, self.sequences)
                    current_cv_value = hmm_model.score(X_test, len_test)
                    if current_cv_value > selected_model_cv_value:  # lower scores are better
                        selected_model = hmm_model
                        selected_model_cv_value = current_cv_value
                    return selected_model

                except Exception as e:
                    return self.base_model(self.n_constant)
