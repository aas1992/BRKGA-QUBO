import abc
from dataclasses import dataclass
from typing import List

import numpy as np

import brkga


@dataclass
class Criterion(abc.ABC):
    @abc.abstractmethod
    def is_met(self, brkga_obj: brkga.BRKGA):
        raise NotImplementedError()


@dataclass
class MaxGenerations(Criterion):
    max_generations: int

    def is_met(self, brkga_obj: brkga.BRKGA):
        return brkga_obj.generation > self.max_generations


@dataclass
class GenerationsWithNoImprovment(Criterion):
    generations_no_improvement: int

    def is_met(self, brkga_obj: brkga.BRKGA):
        return brkga_obj.generations_no_improvement == self.generations_no_improvement


@dataclass
class PercentageDeviation(Criterion):
    opt: float
    data: dict = None

    def __post_init__(self):
        self.data = self.data or {}

    def is_met(self, brkga_obj: brkga.BRKGA):
        try:
            solh = self.data["eval"](brkga_obj.best_ind.genes)
        except KeyError:
            solh = brkga_obj.best_ind.fitness

        if not brkga_obj.maximize:
            return (solh - self.opt) / np.abs(self.opt) <= brkga_obj.epsilon
        else:
            return (self.opt - solh) / np.abs(self.opt) <= brkga_obj.epsilon


@dataclass
class MultipleCriterion(Criterion):
    criteria: List[Criterion]

    def is_met(self, brkga_obj: brkga.BRKGA):
        return any(c.is_met(brkga_obj) for c in self.criteria)
