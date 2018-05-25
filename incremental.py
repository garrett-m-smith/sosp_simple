# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:26:04 2018

@author: garrettsmith

Word-by-word SOSP sentence processing

The lexicon is a dictionary where the keys are attachment sites (head and
dependent) and the values are lists the features.

A treelet is a vector of head identities (phon. forms), head features, and
a fixed number of dependent features for each word.

DESIGN CHOICE: To allow for strings shorter than max_sent_length, added EMPTY
lexical item as placeholder. Short but fully linked parses, e.g., the->dog
EMPTY, are fully harmonious.

CHANGE TO PREV. CHOICE (5/24): Only parses with all required attachments are
now fully harmonious. This is done by implementing the at-most-one-hot rule
as a local constraint for each attch. site.

DESIGN CHOICE: A seq. of all EMPTYs has a local harmony of missing_link_penalty

DESIGN CHOICE: Ambiguous words are disambiguated in the lexicon file, but if
they share a phonological form, only a single phonological form is used for
making the dimension names. This is also how optional attachments can be
handled (although I haven't tried that yet).

DESIGN CHOICE: eliminating link patterns like L_W0_W1_d0 L_W1_W0_d0, i.e.,
"circular" link patterns.

DESIGN CHOICE: init. activ. patterns of ambiguous words are the average of
their ambiguous senses.

DESIGN CHOICE: a seq. of all EMPTYs is only penalized for its missing links

DESIGN CHOICE: When a new word is input, predictions/hallucinations about not-
yet-seen words are erased so that the system it always deflected away from an
attr. instead of immediately being at a, w0 w1 EMPTY (no link) low-harm. attr.

DESIGN CHOICE: Include a full lexicon, but if only want to consider particular
sequences, simply pass a corpus of those sequences.

Later maybe: Info about the expected direction of dependents would reduce the
number of dim. Also, after calculating harmonies, could eliminate very
low-harmony centers to simplify system.

For now at least, don't use root/apex node
"""

import yaml
from itertools import product
from sympy.utilities.iterables import multiset_permutations
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from .dynamics import calc_harmony, iterate, euclid_stop, vel_stop, cheb_stop
import pandas as pd


def gen_nlinks_vectors(nlink_dims, maxlinks):
    lconfigs = []
    for i in range(0, maxlinks+1):
        base = [0]*nlink_dims
        if i > 0:
            base[:i] = [1]*i
        perms = multiset_permutations(base)
        lconfigs.extend([i for i in perms])
    return lconfigs  # chain(*lconfigs)


class Struct(object):
    def __init__(self, lex_file=None, features=None, max_sent_length=10,
                 missing_link_cost=0.01, gamma=0.25,
                 stopping_crit='euclid_stop', corpus=None):
        self.max_sent_length = max_sent_length
        self.ndim_per_position = 0
        # Maximum number of possible dependents; change to be fn. that calc.s
        # after reading in lex.
        self.ndep = 2
        self.max_links = self.max_sent_length - 1
        # Multiplier for missing links
        self.missing_link_cost = missing_link_cost
        self.gamma = gamma
        if stopping_crit == 'vel_stop':
            self.stopping_crit = vel_stop
        elif stopping_crit == 'cheb_stop':
            self.stopping_crit = cheb_stop
        else:
            self.stopping_crit = euclid_stop

        self.tau = 0.01  # Time step for discretized dynamics
        self.max_time = 10000  # Max. number of time steps
        self.noise_mag = 0.0001  # default
        self.tol = 0.05  # Stopping tolerance

        if features is None:
            self.features = ['Det', 'N', 'V', 'sg', 'pl']
            self.nfeatures = len(self.features)
        else:
            self.features = features
            self.nfeatures = len(features)

        if lex_file is not None:
            self.lexicon = self._import_lexicon(lex_file)
            pf = []
            for w in self.lexicon:
                pf.append(self.lexicon[w]['phon_form'])
            self.phon_forms = list(dict.fromkeys(pf))
            self.nwords = len(self.lexicon)
            self.nphon_forms = len(self.phon_forms)
            self.pos_names = self._name_pos_dims()
            self.link_names = self._name_links()
            self.nlinks = len(self.link_names)
            self.dim_names = self.pos_names + self.link_names
            self.ndim = len(self.dim_names)
            self.idx_words = {j: i for i, j in enumerate(self.lexicon.keys())}
            self.idx_phon_feat = slice(0, self.nphon_forms)
            self.idx_phon_dict = {j: i for i, j in enumerate(self.phon_forms)}
            self.idx_head_feat = slice(self.nphon_forms, self.nphon_forms
                                       + self.nfeatures)
            self.idx_links = slice(len(self.pos_names), len(self.dim_names))
            self.word_vecs = self._make_word_vecs()
        else:
            print('No lexicon loaded')
            self.lexicon = dict()
            self.nwords = 0
            self.dim_names = None
            self.ndim = None
        # Working with a corpus
        if corpus is not None:
            disamb = corpus.copy()
            for seq in corpus:
                # Disambiguating words
                for word_nr, word in enumerate(seq):
                    ambig_forms = [w for w in self.lexicon if word in w]
                    if len(ambig_forms) > 1:
                        for amb in ambig_forms:
                            rep = [w if w is not word else amb for w in seq]
                            disamb.append(rep)
#                        del disamb[corpus.index(seq)]
                        del disamb[disamb.index(seq)]
            # Also need to add partial subsequences from seqs in corpus
            full_corp = disamb.copy()
            for seq in disamb:
                for i in range(len(seq)-1):
                    full_corp.append(seq[:i+1] + ['EMPTY']*(len(seq)-i-1))
            corp_tuple = map(tuple, full_corp)
            corp_unique = list(map(list, dict.fromkeys(corp_tuple)))
#            self.seq_names = full_corp
            self.seq_names = corp_unique

    def set_params(self, **kwargs):
        for param, val in kwargs.items():
            setattr(self, param, val)

    def _import_lexicon(self, file):
        with open(file, 'r') as stream:
            lex = yaml.safe_load(stream)
        assert 'EMPTY' in lex.keys(), 'Lexicon must include EMPTY.'
        return lex

    def _make_word_vecs(self):
        """Builds word vecs, return them in a NumPy array
        """
        word_list = []
        for word in self.lexicon:
            curr = []
            word_phon = self.lexicon[word]['phon_form']
            phon = [0.] * self.nphon_forms
            phon[self.idx_phon_dict[word_phon]] = 1.0
            curr.extend([i for i in phon])
            curr.extend(self.lexicon[word]['head'])
            if self.lexicon[word]['dependents'] is None:
                curr.extend([-1.] * self.ndep * self.nfeatures)
            else:
                for dep in self.lexicon[word]['dependents']:
                    curr.extend(self.lexicon[word]['dependents'][dep])
                ndeps = len(self.lexicon[word]['dependents'])
                if ndeps > 0:
                    # Code non-existent features as -1s as placeholders
                    curr.extend([-1.] * (self.ndep-ndeps) * self.nfeatures)
            word_list.append(curr)
        return np.array(word_list)

    def _name_seqs(self):
        """Finds all word sequences up to max_sentence_lengths. The centers
        will be these with allowed link combinations appended (done later).
        """
        # One approach: for each possible sequence of words, find all allowed
        # feature/link combinations.
        if self.seq_names:
            word_seqs = self.seq_names
        else:
            non_empty = {k: self.lexicon[k] for k in self.lexicon
                         if k not in 'EMPTY'}
            # For storing all possible sequences of words
            word_seqs = []
            # Manually adding the empty sequence
            word_seqs.append(['EMPTY'] * self.max_sent_length)
            for i in range(self.max_sent_length):
                pr = product(non_empty, repeat=i+1)
                word_seqs.extend([list(x) for x in pr])
            for i in range(len(word_seqs)):
                curr_len = len(word_seqs[i])
                if curr_len < self.max_sent_length:
                    word_seqs[i].extend(['EMPTY'] * (self.max_sent_length
                                                     - curr_len))
        return word_seqs

    def _make_seq_vecs(self):
        """Returns a list of sequence vectors in which each element holds word
        vectors concatenated together.
        """
        word_vec = self._make_word_vecs()
        seq_names = self._name_seqs()
        seq_vecs = []
        for seq in seq_names:
            curr_seq = []
            for word in seq:
                curr_word = self.idx_words[word]
                curr_seq.extend(word_vec[curr_word])
            seq_vecs.append(curr_seq)
        self.seq_vecs = seq_vecs
        return seq_vecs

    def _prune_links(self):
        """Returns an array of link vectors after removing the ones disallowed
        under the constraints of SOSP
        """
        link_names = self._name_links()
        nlink_dims = len(link_names)
        link_vecs = gen_nlinks_vectors(nlink_dims, self.max_links)
        # A little kludgy, but works for now...
        if self.max_sent_length == 2:
            return(link_vecs)
        to_rm = []
        to_keep = []
        for i, lvec in enumerate(link_vecs):
            # Remove vectors that have the same word attached twice as a dep.
            for word_nr in range(self.max_sent_length):
                dim_per_word = self.ndep * (self.max_sent_length-1)
                init = word_nr*dim_per_word
                idx = slice(init, init+dim_per_word)
                if sum(lvec[idx]) >= self.max_links:
                    to_rm.append(i)
#                    to_keep.append(lvec)
                # Next, rm vectors with more than one thing attached to the
                # same dep attch site.
                for dep in ['d0', 'd1']:
                    word_str = 'W' + str(word_nr) + '_' + dep
                    dep_idx = [j for j, w in enumerate(link_names)
                               if word_str in w]
                    if sum([lvec[k] for k in dep_idx]) >= self.max_links:
                        to_rm.append(i)
            # Now rm links that form cycles
            for wn in range(self.max_sent_length-1):
                w0 = wn
                w1 = wn + 1
                for d in ['d' + str(j) for j in range(self.ndep)]:
                    s0 = '_'.join(['L', 'W' + str(w0), 'W' + str(w1), d])
                    idx0 = link_names.index(s0)
                    s1 = '_'.join(['L', 'W' + str(w1), 'W' + str(w0), d])
                    idx1 = link_names.index(s1)
                    if lvec[idx0] == 1 and lvec[idx1] == 1:
                        to_rm.append(i)
            # Finally, remove links that aren't possible with the vocabulary
        return [link_vecs[k] for k in range(len(link_vecs)) if k not in to_rm]

    def _name_links(self):
        print('Naming links...')
        links = []
        for pos_nr in range(self.max_sent_length):
            other_positions = [x for x in range(self.max_sent_length)
                               if x != pos_nr]
            # Any word can appear at any position, so use whole lexicon here
            for op in other_positions:
                for dep in ['d0', 'd1']:  # first and second dependents
                    links.append('_'.join(['L', 'W' + str(pos_nr),
                                           'W' + str(op), dep]))
        return links

    def _name_pos_dims(self):
        """Returns a list of the dimension names. There are always ndep
        dependents at a position regardless of what word is in that position.
        Also only creates one phonological form for ambiguous words, like
        'the_sg' and 'the_pl.'
        """
        assert self.lexicon is not None, 'Must initialize lexicon.'
        print('Naming position dimensions...')
        per_position = []
        for word in self.phon_forms:
            per_position.append(word)
        for feat in self.features:
            per_position.append(feat)
        for dep in range(self.ndep):
            for feat in self.features:
                per_position.append('d' + str(dep) + '_' + feat)
        self.ndim_per_position = len(per_position)

        all_names = []
        for i in range(self.max_sent_length):
            tmp = ['W' + str(i) + '_' + pf for pf in per_position]
            for x in tmp:
                all_names.append(x)
        return all_names

    def gen_centers(self):
        """Will return a NumPy array with a center on each row.

        Because links are only care about sentence position and attch. site,
        don't have to worry about what words are in the positions, except to
        make sure they allow dependents.

        Note: need to create 2 different centers when there's a 0.5 in the vec
        """
        # Notes: link vec of zeros is always possible, no matter how many words
        # have been input. No links turned on after reading first word.
        # As words come in, can only allow centers with them attching somehow
        # to previous words, not looking ahead.
        seq_vecs = self._make_seq_vecs()
        seq_names = self._name_seqs()
        assert len(seq_vecs) == len(seq_names), \
            'Number of sequence vectors mismatches number of sequence names.'
        link_names = self._name_links()
        link_vecs = self._prune_links()
        centers = []
        # Cycle through seqs and find allowed links
        for seq_name, seq in zip(seq_names, seq_vecs):
            curr_seq = seq.copy()
            if seq_name[0] == 'EMPTY':
                # Assumes 0th link vec is one with no links!
                centers.append(curr_seq + link_vecs[0])
            elif seq_name[1] == 'EMPTY':
                centers.append(curr_seq + link_vecs[0])
            else:
                # Need to exclude attchs. to EMPTYs
                try:
                    first_empty = seq_name.index('EMPTY')
                    empties = ['W' + str(i) for i in
                               range(first_empty, self.max_sent_length)]
                    # Indexing the dimensions that have links to EMPTYs
                    empty_idx = [i for i, ln in enumerate(link_names) for e in
                                 empties if e in ln]
                except ValueError:
                    empty_idx = []
                to_rm = []
                for lconfig in link_vecs:
                    for i in empty_idx:
                        if lconfig[i] != 0:
                            to_rm.append(lconfig)
                # Now removing link configs if they link to a non-existent
                # dependent
                for word_nr, word in enumerate(seq_name):
                    if self.lexicon[word]['dependents'] is None:
                        null_attch = ['W' + str(word_nr) + '_' + 'd'
                                      + str(j) for j in range(self.ndep)]
                        null_idx = [i for i, ln in enumerate(link_names)
                                    for n in null_attch if n in ln]
                        for lconfig in link_vecs:
                            for i in null_idx:
                                if lconfig[i] != 0:
                                    to_rm.append(lconfig)
                    elif len(self.lexicon[word]['dependents']) < self.ndep:
                        null_attch = ['W' + str(word_nr) + '_' + 'd'
                                      + str(j) for j in
                                      range(1, self.ndep)]
                        null_idx = [i for i, ln in enumerate(link_names)
                                    for n in null_attch if n in ln]
                        for lconfig in link_vecs:
                            for i in null_idx:
                                if lconfig[i] != 0:
                                    to_rm.append(lconfig)
                    # Now, removing links to/from EMPTYs
                    if word == 'EMPTY':
                        idx = [i for i, ln in enumerate(link_names)
                               if 'W' + str(word_nr) in ln]
                        for lconfig in link_vecs:
                            if any([lconfig[j] for j in idx]):
                                to_rm.append(lconfig)
                # Finally, removing any link configs w/ multiple links to
                # same attch. site
                    for lconfig in link_vecs:
                        mult_gov = [l for l in link_names if 'L_W' +
                                    str(word_nr) in l]
                        idx = [i for i, ln in enumerate(self.link_names) if ln
                               in mult_gov]
                        if sum([lconfig[i] for i in idx]) >= 2:
                            to_rm.append(lconfig)
                # Excluding to_rm
                configs_to_use = [c for c in link_vecs if c not in to_rm]
                for config in configs_to_use:
                    centers.append(curr_seq + config)
        # Getting rid of duplicates
        ctuple = map(tuple, centers)
        centers_unique = list(dict.fromkeys(ctuple))
        centers_array = np.array(centers_unique)
        centers_array[centers_array < 0] = 0.0  # Getting rid of -1s
        self.centers = centers_array
        print('Number of centers generated: {}'.format(centers_array.shape[0]))
        return

    def which_nonzero(self, center):
        """Returns the names of the dimensions in a cetner that are non-zero.
        """
        idx = list(np.where(center != 0)[0])
        return [self.dim_names[i] for i in idx]

    def look_up_center(self, active):
        """Returns the center (if it exists) that corresponds to the given
        dimensions being active.
        """
        idx = [self.dim_names.index(dim) for dim in active]
        test_vec = [0] * len(self.dim_names)
        for i in idx:
            test_vec[i] = 1
        if test_vec in self.centers.tolist():
            lidx = np.where((self.centers == test_vec).all(axis=1))
            lharmony = self.local_harmonies[lidx]
            print('Local harmony: {}\nCenter number: {}\nCenter: {}'.format(
                    lharmony, lidx, test_vec))
        else:
            print('Active dimensions don\'t correspond to a center.')
        return

    def hamming_dist(self, vec0, vec1):
        return sum(f0 != f1 for f0, f1 in zip(vec0, vec1))

    def feat_match(self, vec0, vec1):
        assert len(vec0) == len(vec1), 'Feature vectors not of equal length'
        return 1 - (self.hamming_dist(vec0, vec1) / len(vec0))

    def calculate_local_harmonies(self):
        """Cycle through the centers and use self.lexicon to look up features.
        """
        local_harmonies = np.ones(self.centers.shape[0])
        for c, center in enumerate(self.centers):
            # Find which dims are active
            nonzero = self.which_nonzero(self.centers[c])
            # Getting active links from there
            active_links = [nonzero[i] for i, dim in enumerate(nonzero)
                            if 'L_' in dim]
            nempties = len([dim for dim in nonzero if 'EMPTY' in dim])
            if nempties == self.max_sent_length:
                # This is a choice:
                local_harmonies[c] *= self.missing_link_cost**self.max_links
#                pass
                continue
            for link in active_links:
                # get locations of feat vecs
                _, dep_word_nr, head_word_nr, head_dep = link.split('_')
                # Just the position number
                dep_nr = int(dep_word_nr[1])
                dep_slice = slice(dep_nr * self.ndim_per_position
                                  + self.nphon_forms,
                                  dep_nr * self.ndim_per_position
                                  + self.nphon_forms + self.nfeatures)
                # Head features of the dependent treelet
                v0 = center[dep_slice]
                # Getting features of dependent attch. site on the head
                head_str = '_'.join([head_word_nr, head_dep])
                tmp = [i for i, x in enumerate(self.pos_names) if head_str
                       in x]
                head_slice = slice(tmp[0], tmp[0] + self.nfeatures)
                v1 = center[head_slice]
                local_harmonies[c] *= self.feat_match(v0, v1)
            # Penalizing missing links
            active_words = [nonzero[i] for i, dim in enumerate(nonzero)
                            for ph in self.phon_forms if ph in dim]
            spl = [item.split('_') for item in active_words]
            assert max([len(it) for it in spl]) == 2, 'Error identifying words'
            for pos, word in spl:
                if word == 'EMPTY':
                    continue
                ambig = [ph for ph in self.lexicon if word in ph]
                if len(ambig) > 1:
                    for form in ambig:
                        head_feats = [self.features[i] for i, val in
                                      enumerate(self.lexicon[form]['head'])
                                      if val == 1]
                        hfeat_pos = [pos + '_' + x for x in head_feats]
                        # If this form isn't the one in the center, skip it
                        if not all(x in nonzero for x in hfeat_pos):
                            continue
                        if (self.lexicon[form]['gov_req'] and not
                            any(x for x in active_links if 'L_' + pos + '_W' in x)):
                            local_harmonies[c] *= self.missing_link_cost
                        if self.lexicon[form]['dependents'] is not None:
                            for dep_nr in range(len(self.lexicon[form]['dependents'])):
                                if (self.lexicon[form]['dep_req'][dep_nr] and
                                    not any(x for x in active_links if pos +
                                            '_d' + str(dep_nr) in x)):
                                    local_harmonies[c] *= self.missing_link_cost
                else:
                    if (self.lexicon[word]['gov_req'] and
                        not any(x for x in active_links if 'L_' + pos + '_W'  in x)):
                        local_harmonies[c] *= self.missing_link_cost
                    if self.lexicon[word]['dependents'] is not None:
                        for dep_nr in range(len(self.lexicon[word]['dependents'])):
                            if (self.lexicon[word]['dep_req'][dep_nr] and not
                                any(x for x in active_links if pos + '_d' + str(dep_nr) in x)):
                                local_harmonies[c] *= self.missing_link_cost
            # Old way: across-the-board, top-down penalty for too few links
#            if len(active_links) < self.max_links - nempties:
#                local_harmonies[c] *= (self.missing_link_cost **
#                                       (self.max_links -
#                                        len(active_links)))
        self.local_harmonies = local_harmonies
        return

    def input_word(self, state_vec, word, pos):
        """Inputs a new word at a particular position by overwriting the values
        of the state vector at the relevant positions.
        """
        assert (pos + 1) <= self.max_sent_length, \
            'Can only add words up to max_sent_length'
        # First, get the feature vector(s) from the lexicon
        ambig_words = [w for w in self.lexicon if word in w]
        # Then, average them in case the word is ambiguous
        word_vec = np.zeros(self.nfeatures)
        for w in ambig_words:
            word_vec += np.array(self.lexicon[w]['head'])
        word_vec /= len(ambig_words)
        # Getting dep. features
        dep_feats = np.zeros(self.ndep * self.nfeatures)
        for i, w in enumerate(ambig_words):
            if self.lexicon[w]['dependents'] is not None:
                idx = slice(i*self.nfeatures, i*self.nfeatures+self.nfeatures)
                for d in self.lexicon[w]['dependents']:
                    # Getting avg. of deps in case the word has multiple senses
                    dep_feats[idx] += np.array(self.lexicon[w]['dependents'][d])
        dep_feats /= len(ambig_words)
        # Finally, turn on the averaged features at the correct possition
        phon = np.zeros(self.nphon_forms)
        phon[self.idx_phon_dict[word]] = 1.0
        whole_vec = np.zeros(self.ndim_per_position * (self.max_sent_length
                                                       - pos))
        whole_vec[:self.nphon_forms] = phon
        whole_vec[self.nphon_forms:self.nphon_forms+self.nfeatures] = word_vec
        whole_vec[self.nphon_forms+self.nfeatures:
                  self.nphon_forms+self.nfeatures+self.ndep*self.nfeatures] \
            = dep_feats
        updated_state = state_vec.copy()
        start = pos*self.ndim_per_position
        stop = self.ndim - self.nlinks
        idx = slice(start, stop)
        updated_state[idx] = whole_vec
        return updated_state

    def neg_harmony(self, x, centers, local_harmonies, gamma):
        return -1 * calc_harmony(x, centers, local_harmonies, gamma)

    def jac_neg_harmony(self, x, centers, local_harmonies, gamma):
        return -1 * iterate(x, centers, local_harmonies, gamma)

    def locate_attrs(self):
        """Finds actual locations of attractors in the full harmony landscape
        using the Newton-CG algorithm on the negative of the harmony fn.
        """
        attrs = np.zeros(self.centers.shape)
        for c in range(self.centers.shape[0]):
            extremum = minimize(self.neg_harmony, self.centers[c],
                                args=(self.centers, self.local_harmonies,
                                      self.gamma), method='L-BFGS-B',
                                jac=self.jac_neg_harmony)
            attrs[c] = extremum.x
        unique_attrs = np.unique(np.round(attrs, 6), axis=0)
        self.attrs = unique_attrs
        print('Found {} unique attractors from {} centers'.format(
                self.attrs.shape[0], self.centers.shape[0]))
        return

    def _zero_state_hist(self):
        self.state_hist = np.zeros((self.max_time, self.ndim))

    def single_run(self, seq=None):
        """Run the model once until stopping criterion is met or
        time runs out.
        """
        assert seq is not None, 'Must provide a sequence of words.'
        self._zero_state_hist()
        self.harmony = np.zeros(self.max_time)
        data = []
        # Input the first word
        curr_pos = 0
        self.state_hist[0, ] = self.input_word(self.state_hist[0, ],
                                               seq[curr_pos], curr_pos)
        # Pre-generate the noise for speed
        noise = (np.sqrt(2 * self.noise_mag * self.tau)
                 * np.random.normal(0, 1, self.state_hist.shape))
        t = 0
        word_t = 0  # for keeping track of max amt. of time ea. word can get
        while t < self.max_time-1:
            not_close = self.stopping_crit(self.state_hist[t], self.attrs,
                                           self.tol)
            if not_close:
                self.state_hist[t+1, ] = (self.state_hist[t, ]
                                          + self.tau *
                                          iterate(self.state_hist[t, ],
                                                  self.centers,
                                                  self.local_harmonies,
                                                  self.gamma)
                                          + noise[t, ])
                self.harmony[t] = calc_harmony(self.state_hist[t, ],
                                               self.centers,
                                               self.local_harmonies,
                                               self.gamma)
                t += 1
                word_t += 1
            else:
                data.append([curr_pos, seq[curr_pos], word_t])
                try:
                    curr_pos += 1
                    self.state_hist[t+1, ] = (self.input_word(
                                              self.state_hist[t, ],
                                              seq[curr_pos], curr_pos))
                    self.harmony[t] = calc_harmony(self.state_hist[t, ],
                                                   self.centers,
                                                   self.local_harmonies,
                                                   self.gamma)
                    t += 1
                    word_t = 0
                except:
                    trunc = self.state_hist[~np.all(self.state_hist == 0,
                                                    axis=1)]
                    return trunc[-1], data
        trunc = self.state_hist[~np.all(self.state_hist == 0, axis=1)]
        return trunc[-1], data

    def many_runs(self, n_runs=100, seq=None):
        """Do repeated Monte Carlo runs. Returns a Pandas data frame with the
        center number and settling time.
        """
        print('Run number:')
        data_list = []
        for run in range(n_runs):
            curr_data = []
            if run % (n_runs // 10) == 0:
                print('[{}] '.format(run), end='')
            final_st, trial_data = self.single_run(seq)
            for w in trial_data:
                curr_data.append(w)
            final_rounded = np.rint(final_st)
            final_rounded += 0.  # getting rid of negative zeros from rounding
#            t = self.state_hist[~np.all(self.state_hist == 0, axis=1)].shape[0]
            for center in range(self.centers.shape[0]):
                if np.all(final_rounded == self.centers[center,]):
#                    data_list.append(trial_data.extend([run, center, t]))
                    to_append = [it + [run, center] for it in curr_data]
                    for it in to_append:
                        data_list.append(it)
        return pd.concat([pd.DataFrame([i], columns=('WordNr', 'Word',
                                                     'WordRT', 'RunNr',
                                                     'FinalCenterNr'))
                          for i in data_list])

    def plot_trace(self):
        trunc = self.state_hist[~np.all(self.state_hist == 0, axis=1)]
        plt.plot(trunc)
        plt.xlabel('Time')
        plt.ylabel('Activation')
        plt.title('Evolution of state vector')
        plt.show()

    def plot_harmony(self):
        trunc = self.harmony[self.harmony != 0]
        plt.plot(trunc)
        plt.xlabel('Time')
        plt.ylabel('Harmony')
        plt.title('Harmony over time')
        plt.show()

    def plot_links(self):
        trunc = self.state_hist[~np.all(self.state_hist == 0, axis=1),
                                -self.nlinks:]
        for dim, ln in zip(range(self.nlinks), self.link_names):
            plt.plot(trunc[:, dim], label=ln)
        plt.xlabel('Time')
        plt.ylabel('Activation')
        plt.title('Link strengths')
        plt.legend()
        plt.show()


if __name__ == '__main__':
#    file = '../test.yaml'
    file = '../../Dissertation/Models/LCLexicon.yaml'
    sent_len = 4
#    corp = [['the', 'dog'], ['an', 'cat']]
#    corp = [['the', 'dog', 'eats'], ['an', 'cat', 'sees']]
    corp = [['smiled', 'at', 'player', 'thrown']]
#    corp = [['the', 'dog', 'eats'],
#            ['an', 'cat', 'eats'],
#            ['dog', 'dog', 'eats']]
#    corp = [['dog', 'sees', 'the', 'cat']]
#    corp = [['the', 'dog', 'sees', 'the', 'cat']]
    # Missing link cost seems to need to be not too small, otherwise it can't
    # get to the attractors with EMPTYs for not-yet-seen words
#    sys = Struct(lex_file=file, features=None, max_sent_length=sent_len,
#                 missing_link_cost=0.5, gamma=0.45,
#                 stopping_crit='cheb_stop', corpus=corp)
    sys = Struct(lex_file=file, features=['N', 'Prep', 'MainVerb',
                                          'Participle'],
                 max_sent_length=sent_len,
                 missing_link_cost=0.5, gamma=0.45,
                 stopping_crit='cheb_stop', corpus=corp)
    sys.gen_centers()
    sys.calculate_local_harmonies()
    sys.locate_attrs()
    sys.set_params(noise_mag=0.00005)
#    final, data = sys.single_run(['an', 'cat'])
#    final, data = sys.single_run(['the', 'dog'])
#    final, data = sys.single_run(['dog', 'eats'])
#    final, data = sys.single_run(['the', 'dog', 'eats'])
#    final, data = sys.single_run(['dog', 'dog', 'eats'])
#    final, data = sys.single_run(['an', 'cat', 'eats'])
#    final, data = sys.single_run(['dog', 'sees', 'the', 'cat'])
#    final, data = sys.single_run(['the', 'dog', 'sees', 'the', 'cat'])
    final, data = sys.single_run(corp[0])
    sns.distplot(sys.local_harmonies, kde=False, rug=True)
    plt.title('Distribution of $h_i$')
    plt.show()
    sys.plot_trace()
    sys.plot_links()
    sys.plot_harmony()
    print(sys.which_nonzero(np.round(final)))
    print(data)
    sys.look_up_center(sys.which_nonzero(np.round(final)))
#    mc = sys.many_runs(10, corp[0])
#    print('\n', mc.groupby(['WordNr']).agg({'WordRT': ['mean', 'std', 'min',
#                                                       'max']}))

    # Saving data:
#    import pickle
#    with open('sosp_test_5word.pkl', 'wb') as output:
#        pickle.dump(sys, output, -1)

    # Importing data:
#    with open('sosp_test_5word.pkl', 'rb') as input:
#        sys = pickle.load(input)
