from abc import ABC, abstractmethod

import numpy as np
import scipy.stats
import scipy.stats


def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    return scipy.stats.entropy(counts)


def weighted_entropy(x, s, weighting_fun):
    if weighting_fun == 'mean':
        wf = np.mean
    elif weighting_fun == 'hmean':
        wf = scipy.stats.hmean
    elif weighting_fun == 'invmean':
        wf = lambda v: np.mean(v) ** -1
    elif weighting_fun == 'invhmean':
        wf = lambda v: scipy.stats.hmean(v) ** -1
    elif weighting_fun == 'distentropy':
        wf = lambda v: scipy.stats.entropy(v ** -1 / (np.sum(v ** -1)))
    label, counts = np.unique(x, return_counts=True)
    scores = [wf(s[x == i]) for i in label]
    counts = counts / np.sum(counts, keepdims=True)
    ent = -np.sum(scores * counts * np.log(counts))
    return ent


class PatchScorer(ABC):

    @staticmethod
    def get_background_mask(saliency_scores, patch_scores, thresh):
        return (saliency_scores < thresh) & (patch_scores < thresh)

    @abstractmethod
    def score(self, patch_scores: np.ndarray, saliency_scores: np.ndarray, patch_label: np.ndarray) -> (float, int):
        pass

    def __init__(self, threshold=-np.inf):
        self.threshold = threshold


class StandardScorer(PatchScorer):
    @property
    @abstractmethod
    def scoring_fun(self):
        return lambda x: x

    def score(self, patch_scores: np.ndarray, saliency_scores: np.ndarray, patch_label: np.ndarray):
        background = self.get_background_mask(patch_scores, saliency_scores, thresh=self.threshold)
        if np.sum(~background) == 0:
            return self.max_score, 0
        valid_patch_scores = patch_scores[~background]
        image_score = self.scoring_fun(valid_patch_scores)
        label = patch_label[np.argmin(patch_scores)]
        return image_score, label


class EntropyScorer(PatchScorer):

    def score(self, patch_scores: np.ndarray, saliency_scores: np.ndarray, patch_label: np.ndarray):
        background = self.get_background_mask(patch_scores, saliency_scores, thresh=self.threshold)
        if np.sum(~background) == 0:
            return self.max_score, 0
        valid_patch_scores = patch_label[~background]
        image_score = entropy(valid_patch_scores)
        label, counts = np.unique(valid_patch_scores, return_counts=True)
        label = label[np.argmax(counts)]
        return image_score, label


class WeightedEntropyScorer(PatchScorer):
    def score(self, patch_scores: np.ndarray, saliency_scores: np.ndarray, patch_label: np.ndarray,
              weighting='mean'):
        background = self.get_background_mask(patch_scores, saliency_scores, thresh=self.threshold)
        if np.sum(~background) == 0:
            return self.max_score, 0
        valid_patch_label = patch_label[~background]
        valid_scores = patch_scores[~background]
        valid_scores = valid_scores
        image_score = weighted_entropy(valid_patch_label, valid_scores, weighting_fun=weighting)
        label, counts = np.unique(valid_patch_label, return_counts=True)
        label = label[np.argmax(counts)]
        return image_score, label


class WeightedEntropyScorer2(PatchScorer):
    def score(self, patch_scores: np.ndarray, saliency_scores: np.ndarray, patch_label: np.ndarray,
              weighting='mean'):
        background = self.get_background_mask(patch_scores, saliency_scores, thresh=self.threshold)
        if np.sum(~background) == 0:
            return self.max_score, 0
        valid_patch_label = patch_label[~background]
        valid_scores = patch_scores[~background]
        label, counts = np.unique(valid_patch_label, return_counts=True)
        frequencies = np.zeros(np.max(patch_label) + 1)
        frequencies[label] = counts / np.sum(counts, keepdims=True)
        patch_surprise = frequencies[valid_patch_label]
        ent = -np.mean(valid_scores * np.log(patch_surprise))
        return ent, np.argmax(counts)


class MinScorer(StandardScorer):
    @property
    def scoring_fun(self):
        return np.min


class MaxScorer(StandardScorer):
    @property
    def scoring_fun(self):
        return np.max


class MeanScorer(StandardScorer):
    @property
    def scoring_fun(self):
        return np.mean
