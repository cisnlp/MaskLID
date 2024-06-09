import fasttext
import numpy as np
import re
import string
from copy import deepcopy

class MaskLID:
    """A class for code-switching language identification using iterative masking."""
    
    def __init__(self, model_path, languages=-1):
        """Initialize the MaskLID class.
        
        Args:
            model_path (str): The path to the fastText model.
            languages (int or list, optional): The indices or list of language labels to consider. Defaults to -1.
        """
        self.model = fasttext.load_model(model_path)
        self.output_matrix = self.model.get_output_matrix()
        self.labels = self.model.get_labels()
        self.language_indices = self._compute_language_indices(languages)
        self.labels = [self.labels[i] for i in self.language_indices]

    def _compute_language_indices(self, languages):
        """Compute indices of selected languages.
        
        Args:
            languages (int or list): The indices or list of language labels.
            
        Returns:
            list: Indices of selected languages.
        """
        if languages != -1 and isinstance(languages, list):
            return [self.labels.index(l) for l in set(languages) if l in self.labels]
        return list(range(len(self.labels)))

    def _softmax(self, x):
        """Compute softmax values for each score in array x.
        
        Args:
            x (numpy.ndarray): Input array.
            
        Returns:
            numpy.ndarray: Softmax output.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _normalize_text(self, text):
        """Normalize input text.
        
        Args:
            text (str): Input text.
            
        Returns:
            str: Normalized text.
        """
        replace_by = " "
        replacement_map = {ord(c): replace_by for c in '\n_:' + '•#{|}' + string.digits}
        text = text.translate(replacement_map)
        return re.sub(r'\s+', ' ', text).strip()

    def predict(self, text, k=1):
        """Predict the language of the input text.
        
        Args:
            text (str): Input text.
            k (int, optional): Number of top predictions to retrieve. Defaults to 1.
            
        Returns:
            tuple: Top predicted labels and their probabilities.
        """
        sentence_vector = self.model.get_sentence_vector(text)
        result_vector = np.dot(self.output_matrix, sentence_vector)
        softmax_result = self._softmax(result_vector)[self.language_indices]
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]
        return tuple(top_k_labels), top_k_probs

    def compute_v(self, sentence_vector):
        """Compute the language vectors for a given sentence vector.
        
        Args:
            sentence_vector (numpy.ndarray): Sentence vector.
            
        Returns:
            list: Sorted list of labels and their associated vectors.
        """
        result_vector = np.dot(self.output_matrix[self.language_indices, :], sentence_vector)
        return sorted(zip(self.labels, result_vector), key=lambda x: x[1], reverse=True)

    def compute_v_per_word(self, text):
        """Compute language vectors for each word in the input text.
        
        Args:
            text (str): Input text.
            
        Returns:
            dict: Dictionary containing language vectors for each word.
        """
        text = self._normalize_text(text)
        words = self.model.get_line(text)[0]
        words = [w for w in words if w not in ['</s>', '</s>']]
        subword_ids = [self.model.get_subwords(sw)[1] for sw in words]
        sentence_vector = [np.sum([self.model.get_input_vector(id) for id in sid], axis=0) for sid in subword_ids]

        dict_text = {}
        for i, word in enumerate(words):
            key = f"{i}_{word}"
            dict_text[key] = {'logits': self.compute_v(sentence_vector[i])}

        return dict_text

    def mask_label_top_k(self, dict_text, label, top_keep, top_remove):
        """Mask top predictions for a given label.
        
        Args:
            dict_text (dict): Dictionary containing language vectors for each word.
            label (str): Label to mask.
            top_keep (int): Number of top predictions to keep.
            top_remove (int): Number of top predictions to remove.
            
        Returns:
            tuple: Dictionaries of remaining and deleted words after masking.
        """
        dict_remained = deepcopy(dict_text)
        dict_deleted = {}

        for key, value in dict_text.items():
            logits = value['logits']
            labels = [t[0] for t in logits]

            if label in labels[:top_keep]:
                dict_deleted[key] = dict_remained[key]

            if label in labels[:top_remove]:
                dict_remained.pop(key, None)

        return dict_remained, dict_deleted

    @staticmethod
    def get_sizeof(text):
        """Compute the size of text in bytes.
        
        Args:
            text (str): Input text.
            
        Returns:
            int: Size of text in bytes.
        """
        return len(text.encode('utf-8'))

    @staticmethod
    def custom_sort(word):
        """Custom sorting function for words.
        
        Args:
            word (str): Input word.
            
        Returns:
            int or float: Sorted value.
        """
        match = re.match(r'^(\d+)_', word)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # Return infinity for words without numbers at the beginning

    def sum_logits(self, dict_data, label):
        """Compute the sum of logits for a specific label across all words.
        
        Args:
            dict_data (dict): Dictionary containing language vectors for each word.
            label (str): Label to sum logits for.
            
        Returns:
            float: Total sum of logits for the given label.
        """
        total = 0
        for value in dict_data.values():
            logits = value['logits']
            labels = [t[0] for t in logits]
            if label in labels:
                total += logits[labels.index(label)][1]
        return total

    def predict_codeswitch(self, text, beta, alpha, min_prob, min_length, max_lambda=1, max_retry=3, alpha_step_increase=5, beta_step_increase=5):
        """Predict language switching points in the input text.
        
        Args:
            text (str): Input text.
            beta (int): Number of top predictions to keep.
            alpha (int): Number of top predictions to remove.
            min_prob (float): Minimum probability threshold for language prediction.
            min_length (int): Minimum length of text after masking.
            max_lambda (int, optional): Maximum number of iterations. Defaults to 1.
            max_retry (int, optional): Maximum number of retries. Defaults to 3.
            alpha_step_increase (int, optional): Step increase for alpha. Defaults to 5.
            beta_step_increase (int, optional): Step increase for beta. Defaults to 5.
        Returns:
            dict: Predicted language switching points and associated information.
        """
        info = {}
        index = 0
        retry = 0

        # compute v
        dict_data = self.compute_v_per_word(text)

        while index < max_lambda and retry < max_retry:
            
            # predict the text
            pred = self.predict(text, k=1)
            label = pred[0][0]
            
            # save the current text in case of step back
            prev_text = text
            # mask
            dict_data, dict_masked = self.mask_label_top_k(dict_data, label, beta, alpha)

            # get the text from the masked text and remained text
            masked_text = ' '.join(x.split('_', 1)[1] for x in dict_masked.keys())
            text = ' '.join(x.split('_', 1)[1] for x in dict_data.keys())
            
            # save info
            if self.get_sizeof(masked_text) > min_length or index == 0:
                temp_pred = self.predict(masked_text)

                if (temp_pred[1][0] > min_prob and temp_pred[0][0] == label) or index == 0:
                    info[index] = {
                        'label': label,
                        'text': masked_text,
                        'text_keys': dict_masked.keys(),
                        'size': self.get_sizeof(masked_text),
                        'sum_logit': self.sum_logits(dict_masked, label)
                    }
                    index += 1
                else:
                    text = prev_text
                    beta += beta_step_increase
                    alpha += alpha_step_increase
                    retry += 1
            else:
                text = prev_text
                beta += beta_step_increase
                alpha += alpha_step_increase
                retry += 1

            if self.get_sizeof(text) < min_length:
                break

        
        # post-process 
        post_info = {}
        for value in info.values():
            key = value['label']
            if key in post_info:
                post_info[key].extend(value['text_keys'])
            else:
                post_info[key] = list(value['text_keys'])

        # join sorted the text from list of keys
        for key in post_info:
            post_info[key] = ' '.join([x.split('_', 1)[1] for x in sorted(set(post_info[key]), key=self.custom_sort)])
                
                
        return post_info
