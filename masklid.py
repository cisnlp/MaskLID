import contextlib
import copy
import re
import string

import unicodedata

import fasttext
import numpy as np

try:
    from threadpoolctl import ThreadpoolController
except ImportError:  # optional: without it NumPy uses its default number of BLAS threads
    ThreadpoolController = None

_CONTROLLER = None

# Recommended settings, tuned on the dev split of cis-lmu/cs-lid-eval with GlotLID v4 and all its labels.
# BALANCED: the best exact match over code-switched and monolingual sentences; CAUTIOUS: fewer false code-switches
# on monolingual text. Use as masklid_model.predict_codeswitch(text, **MASKLID_BALANCED).
MASKLID_BALANCED = dict(beta=20, alpha=5, max_lambda=3, min_length=1, min_prob=0.3, max_retry=3,
                        alpha_step_increase=3, beta_step_increase=5, assign="switch", switch_penalty=4.0, min_chars=6)
MASKLID_CAUTIOUS = dict(beta=20, alpha=3, max_lambda=3, min_length=10, min_prob=0.9, max_retry=3,
                        alpha_step_increase=3, beta_step_increase=5, assign="switch", switch_penalty=4.0, min_chars=6)


# Numbers, punctuation and symbols belong to no language (Unicode categories N*, P*, S*): they are split off at the
# start and end of a word, never used to identify a language, and shown without a label.
import unicodedata

def is_neutral(ch):
    return unicodedata.category(ch)[0] in "NPS"


def _is_number(ch):
    return unicodedata.category(ch)[0] == "N"


def _split_neutral(s):
    """Cut a run of neutral characters into number tokens and punctuation tokens."""
    out, cur, cur_num = [], "", None
    for i, ch in enumerate(s):
        num = _is_number(ch) or (cur_num and 0 < i < len(s) - 1 and _is_number(s[i + 1]))  # 3.5, 1,000, 12:30
        if cur and num != cur_num:
            out.append(cur)
            cur = ""
        cur, cur_num = cur + ch, num
    if cur:
        out.append(cur)
    return out


def tokenize(text):
    """[(token, neutral)] in text order; white space separates chunks and is dropped."""
    tokens = []
    for chunk in text.split():
        idx = [i for i, ch in enumerate(chunk) if not is_neutral(ch)]
        if not idx:
            tokens += [(t, True) for t in _split_neutral(chunk)]
            continue
        a, b = idx[0], idx[-1] + 1
        tokens += [(t, True) for t in _split_neutral(chunk[:a])]
        tokens.append((chunk[a:b], False))
        tokens += [(t, True) for t in _split_neutral(chunk[b:])]
    return tokens


def language_text(text):
    """The words of the text without the numbers, punctuation and symbols around them, one space apart."""
    return " ".join(t for t, n in tokenize(text) if not n)



def _single_blas_thread():
    """A context that runs NumPy's BLAS on one thread.

    MaskLID's matrix products are small (words x labels); a multithreaded BLAS spends much
    longer starting threads than computing them, up to 20x on many-core machines. The
    limit applies inside MaskLID calls only.
    """
    global _CONTROLLER
    if ThreadpoolController is None:
        return contextlib.nullcontext()
    if _CONTROLLER is None:
        _CONTROLLER = ThreadpoolController()
    return _CONTROLLER.limit(limits=1, user_api="blas")


class MaskLID:
    """A class for code-switching language identification using iterative masking."""

    def __init__(self, model_path, languages=-1, renormalize=False, cache_size=200000):
        """Initialize the MaskLID class.

        Args:
            model_path (str or fastText model): The path to the fastText model, or a model already loaded with
                fasttext.load_model (it is used as is, not copied).
            languages (int or list, optional): The indices or list of language labels to consider. Defaults to -1.
            renormalize (bool, optional): If True, probabilities are a softmax over the selected languages only, so
                they sum to 1 over them. If False (the default, as in the paper), the softmax is over all labels of
                the model and the selected labels are read from it, so a text in none of the selected languages
                gets low probabilities for all of them.
            cache_size (int, optional): How many word vectors to keep between calls. Defaults to 200000.
        """
        self.model = fasttext.load_model(model_path) if isinstance(model_path, str) else model_path
        self.output_matrix = self.model.get_output_matrix()
        self.all_labels = self.model.get_labels()
        self.renormalize = renormalize
        self.cache_size = cache_size
        self._word_cache = {}
        self._set_languages(languages)

    def _set_languages(self, languages):
        self.language_indices = self._compute_language_indices(languages)
        self.labels = [self.all_labels[i] for i in self.language_indices]

    def with_languages(self, languages, renormalize=None):
        """A MaskLID limited to `languages` that shares the loaded model (and its word cache) with this one.

        Args:
            languages (int or list): As in __init__.
            renormalize (bool, optional): As in __init__; defaults to this object's setting.

        Returns:
            MaskLID: The new object; this one is not changed.
        """
        other = copy.copy(self)
        other._set_languages(languages)
        if renormalize is not None:
            other.renormalize = renormalize
        return other

    def _compute_language_indices(self, languages):
        """Compute indices of selected languages.

        Args:
            languages (int or list): The indices or list of language labels.

        Returns:
            list: Indices of selected languages.
        """
        if languages != -1 and isinstance(languages, list):
            return [self.all_labels.index(l) for l in set(languages) if l in self.all_labels]
        return list(range(len(self.all_labels)))

    def _softmax(self, x):
        """Compute softmax values for each score in array x."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _normalize_text(self, text):
        """Normalize input text: digits and '\\n_:•#{|}' become spaces, white space is squeezed."""
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
        with _single_blas_thread():
            return self._predict(text, k)

    def _predict(self, text, k=1):
        # fastText's sentence vector takes one line; line breaks count as spaces, as in the words of the text
        text = text.replace('\r', ' ').replace('\n', ' ')
        result_vector = np.dot(self.output_matrix, self.model.get_sentence_vector(text))
        if self.renormalize:
            softmax_result = self._softmax(result_vector[self.language_indices])
        else:
            softmax_result = self._softmax(result_vector)[self.language_indices]
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]
        return tuple(top_k_labels), top_k_probs

    def _word_vector(self, word):
        """Sum of the input vectors of a word's features (its id if in the dictionary, and its n-grams)."""
        v = self._word_cache.get(word)
        if v is None:
            ids = self.model.get_subwords(word)[1]
            v = np.sum([self.model.get_input_vector(i) for i in ids], axis=0)
            if len(self._word_cache) >= self.cache_size:
                self._word_cache.clear()
            self._word_cache[word] = v
        return v

    def compute_v_per_word(self, text):
        """Compute language logits for each word in the input text.

        Args:
            text (str): Input text.

        Returns:
            dict: For every word ("index_word"), its labels and logits sorted by logit, highest first.
        """
        words, scores = self._word_scores(text)
        order = np.argsort(-scores, axis=1, kind="stable") if len(words) else scores
        return {"%d_%s" % (i, w): {"logits": [(self.labels[j], scores[i, j]) for j in order[i]]}
                for i, w in enumerate(words)}

    def _word_scores(self, text):
        words = [w for w in self.model.get_line(self._normalize_text(text))[0] if w != '</s>']
        if not words:
            return words, np.zeros((0, len(self.language_indices)))
        vectors = np.stack([self._word_vector(w) for w in words])
        return words, vectors @ self.output_matrix[self.language_indices].T

    def compute_v(self, sentence_vector):
        """Compute the language logits for a given sentence vector.

        Returns:
            list: (label, logit) pairs sorted by logit, highest first.
        """
        result_vector = np.dot(self.output_matrix[self.language_indices, :], sentence_vector)
        return sorted(zip(self.labels, result_vector), key=lambda x: x[1], reverse=True)

    def mask_label_top_k(self, dict_text, label, top_keep, top_remove):
        """Mask top predictions for a given label (on the output of compute_v_per_word).

        Returns:
            tuple: Dictionaries of remaining and deleted words after masking.
        """
        dict_remained = dict(dict_text)
        dict_deleted = {}
        for key, value in dict_text.items():
            labels = [t[0] for t in value['logits']]
            if label in labels[:top_keep]:
                dict_deleted[key] = dict_remained[key]
            if label in labels[:top_remove]:
                dict_remained.pop(key, None)
        return dict_remained, dict_deleted

    def sum_logits(self, dict_data, label):
        """Compute the sum of logits for a specific label across all words."""
        total = 0
        for value in dict_data.values():
            for lab, logit in value['logits']:
                if lab == label:
                    total += logit
                    break
        return total

    @staticmethod
    def custom_sort(word):
        """Sort key of an "index_word" key: its index."""
        match = re.match(r'^(\d+)_', word)
        return int(match.group(1)) if match else float('inf')

    @staticmethod
    def get_sizeof(text):
        """Compute the size of text in bytes."""
        return len(text.encode('utf-8'))

    def predict_codeswitch(self, text, beta, alpha, min_prob, min_length, max_lambda=1, max_retry=3,
                           alpha_step_increase=5, beta_step_increase=5, return_words=False, assign="mask",
                           switch_penalty=4.0, min_chars=6):
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
            return_words (bool, optional): Also return the tokens of the text and, for each, the label it was
                assigned to (None if none; numbers, punctuation and symbols never get one). Defaults to False.
            assign (str, optional): "mask" (default, as in the paper): words get the language that masked them, and
                words no round took stay unassigned. "switch": MaskLID only finds the languages; then every word is
                assigned to one of them, maximizing the words' log-probabilities minus `switch_penalty` for every
                change of language between neighbouring words, and a language is kept if its words have at least
                `min_chars` characters.
            switch_penalty (float, optional): Cost of a language change for assign="switch". Defaults to 4.0.
            min_chars (int, optional): Characters a language needs for assign="switch". Defaults to 6.
        Returns:
            dict: Predicted languages and the words assigned to each; with return_words, a tuple
            (that dict, words, labels of the words).
        """
        with _single_blas_thread():
            if assign not in ("mask", "switch"):
                raise ValueError('assign must be "mask" or "switch"')
            out = self._predict_codeswitch(text, beta, alpha, min_prob, min_length, max_lambda, max_retry,
                                           alpha_step_increase, beta_step_increase, return_words=True)
            if assign == "switch":
                out = self._switch_assign(out, switch_penalty, min_chars)
            return out if return_words else out[0]

    def _word_logp(self, word):
        """Log-probabilities of all labels of the model for a word on its own."""
        key = ("logp", word)
        v = self._word_cache.get(key)
        if v is None:
            s = self.output_matrix @ self.model.get_sentence_vector(word)
            s = s - s.max()
            v = s - np.log(np.exp(s).sum())
            if len(self._word_cache) >= self.cache_size:
                self._word_cache.clear()
            self._word_cache[key] = v
        return v

    def _switch_assign(self, out, penalty, min_chars):
        """Assign every word to one of the languages MaskLID found (all selected languages if it found none), with
        the best sum of word log-probabilities minus `penalty` per change of language (Viterbi). With one penalty for
        every change, a label's best predecessor is itself or the overall best label, so this is O(words x labels)."""
        result, tokens, token_labels = out
        index = {l: i for i, l in enumerate(self.all_labels)}
        allowed = np.array(sorted(index[l] for l in result) if result else self.language_indices)
        pos = [i for i, (t, n) in enumerate(zip(tokens, token_labels)) if not all(is_neutral(c) for c in t)]
        words = [tokens[i] for i in pos]
        if not words:
            return {}, tokens, [None] * len(tokens)
        score = self._word_logp(words[0])[allowed].copy()
        back = []
        for w in words[1:]:
            best = int(np.argmax(score))
            switch = score[best] - penalty
            stay = score >= switch
            back.append(np.where(stay, np.arange(len(score)), best))
            score = np.where(stay, score, switch) + self._word_logp(w)[allowed]
        path = [int(np.argmax(score))]
        for b in reversed(back):
            path.append(int(b[path[-1]]))
        labels = [self.all_labels[int(allowed[i])] for i in reversed(path)]
        chars = {}
        for w, l in zip(words, labels):
            chars[l] = chars.get(l, 0) + len(w)
        keep = {l for l, c in chars.items() if c >= min_chars} or {max(chars, key=chars.get)}
        token_labels = [None] * len(tokens)
        for i, l in zip(pos, labels):
            token_labels[i] = l if l in keep else None
        order = sorted(keep, key=lambda l: min(k for k, x in enumerate(token_labels) if x == l))
        result = {l: ' '.join(t for t, x in zip(tokens, token_labels) if x == l) for l in order}
        return result, tokens, token_labels

    def _predict_codeswitch(self, text, beta, alpha, min_prob, min_length, max_lambda, max_retry,
                            alpha_step_increase, beta_step_increase, return_words=False):
        # Numbers, punctuation and symbols belong to no language: they are split off the words they touch and never
        # reach the model; only the words are predicted, masked and assigned (the output still shows every token).
        tokens = tokenize(text)
        words = [t for t, neutral in tokens if not neutral]
        if not words:
            return ({}, [t for t, _ in tokens], [None] * len(tokens)) if return_words else {}
        text = ' '.join(words)
        # Every word's labels ranked by logit (highest first; ties in label order). Only the
        # ranks that alpha and beta can reach over all retries are needed.
        scores = np.stack([self._word_vector(w) for w in words]) @ self.output_matrix[self.language_indices].T
        depth = max(beta + max_retry * beta_step_increase, alpha + max_retry * alpha_step_increase)
        order = np.argsort(-scores, axis=1, kind="stable")[:, :depth]
        position = {label: i for i, label in enumerate(self.labels)}
        remaining = np.ones(len(words), dtype=bool)

        info = []
        index = 0
        retry = 0
        while index < max_lambda and retry < max_retry:

            # predict the text
            label = self._predict(text, k=1)[0][0]

            # save the current text in case of step back
            prev_text = text
            # mask: words with the label in their top beta are assigned to it, words with it
            # in their top alpha are removed from the text
            p = position[label]
            masked = remaining & (order[:, :beta] == p).any(1)
            remaining = remaining & ~(order[:, :alpha] == p).any(1)
            masked_text = ' '.join(w for w, m in zip(words, masked) if m)
            text = ' '.join(w for w, r in zip(words, remaining) if r)

            # save info
            accepted = False
            if self.get_sizeof(masked_text) > min_length or index == 0:
                temp_pred = self._predict(masked_text)
                accepted = (temp_pred[1][0] > min_prob and temp_pred[0][0] == label) or index == 0
            if accepted:
                info.append((label, np.nonzero(masked)[0]))
                index += 1
            else:
                # step back: the text is restored (the masked words stay removed)
                text = prev_text
                beta += beta_step_increase
                alpha += alpha_step_increase
                retry += 1

            if self.get_sizeof(text) < min_length:
                break

        # post-process: the words of each label, in text order
        post_info = {}
        for label, indices in info:
            post_info.setdefault(label, set()).update(int(i) for i in indices)
        result = {label: ' '.join(words[i] for i in sorted(indices)) for label, indices in post_info.items()}
        if not return_words:
            return result
        word_labels = [None] * len(words)
        for label, indices in info:
            for i in indices:
                if word_labels[i] is None:
                    word_labels[i] = label
        # every token of the input in order; neutral tokens have no label
        labels_iter = iter(word_labels)
        token_labels = [None if neutral else next(labels_iter) for _, neutral in tokens]
        return result, [t for t, _ in tokens], token_labels
