import abc
from abc import ABC

from brkga.brkga import BRKGA, Individual


class SingleObjectiveBRKGA(BRKGA, ABC):
    def __init__(self, *args, **kwargs):
        super(SingleObjectiveBRKGA, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def get_best_ind_from_population(self):
        raise NotImplementedError("Method not implemented!")

    def update_best_ind(self, best_local):
        if self.best_ind is None:
            self.best_ind = best_local
            self.best_ind.generation = self.generation
        else:
            self.best_ind = self.best(best_local, self.best_ind)
            if self.best_ind == best_local:
                self.best_ind.generarion = self.generation

    def best(self, ind1: Individual, ind2: Individual):
        if self.maximize:
            if ind1.fitness > ind2.fitness:
                return ind1
            return ind2
        # If it is a minimization problem
        if ind1.fitness < ind2.fitness:
            return ind1
        return ind2

    def has_improvement(self, previous_fitness: float) -> bool:
        if self.generation == 1:
            return True
        if self.maximize:
            return self.best_ind.fitness > previous_fitness
        return self.best_ind.fitness < previous_fitness
