import re
from string import ascii_lowercase

import torch
from collections import defaultdict
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, build_ctcdecoder
import os


# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""
    EMPTY_IND = 0

    def _lm_and_vocab_tolower(self, pretrained_lm_path, lower_lm_path, vocab_path):
        unigrams = []
        with open(vocab_path) as f:
            for char in f.read().strip().split("\n"):
                unigrams.append(char.lower())

        if os.path.exists(lower_lm_path):
            return unigrams
        with open(pretrained_lm_path, 'r') as f1:
                with open(lower_lm_path, ) as f2:
                    for line in f1:
                        f2.write(line.lower())
        return unigrams
        

    def __init__(self, alphabet=None, beam_size=None, pretrained_lm_path=None, vocab_path=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.beamsearch_decoder = None
        if pretrained_lm_path is not None:
            lower_lm_path = 'lower-3-gram.pruned.1e-7.arpa'
            unigrams = self._lm_and_vocab_tolower(pretrained_lm_path, lower_lm_path, vocab_path)

            self.beamsearch_decoder = build_ctcdecoder(
                labels=self.vocab,
                kenlm_model_path=lower_lm_path,
                unigrams=unigrams,
            )
            self.beam_size = beam_size
        else:
            if beam_size is not None:
                self.beamsearch_decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)
                self.beam_size = beam_size

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.EMPTY_IND
        for ind in inds:
            if last_char_ind == ind:
                continue
            if ind != self.EMPTY_IND:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind

        return "".join(decoded)
    
    def ctc_beam_search_decode(self, probs):
        probs = probs.detach().cpu().numpy()
        return self.beamsearch_decoder.decode(probs, self.beam_size)
    
    def ctc_beam_search_decode_our(self, log_probs):
        probs = torch.exp(log_probs)
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for prob in probs:
            dp = self._expand_and_merge_path(dp, prob)
            dp = self._truncate_paths(dp, self.beam_size)
        dp = [
            (prefix, proba)
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]
        return dp[0][0]

    def _expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def _truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
