import abc
import time
import numpy as np
from abc import ABC
from itertools import chain
from operator import attrgetter
from random import choice, random
from typing import Union, List

from brkga.brkga import Individual
from brkga.single import SingleObjectiveBRKGA


class DefaultBRKGA(SingleObjectiveBRKGA, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population: Union[List[Individual], List[None]] = self.k * [None]

    @abc.abstractmethod
    def decoder(self, ind, **kwargs):
        """
           Decoder: Function responsible for transforming the vector of
           random keys into a solution for the specific problem.
           It maps a solution to the problem from the vector.
        """
        raise NotImplementedError("Method not implemented!")

    def create_individuals(self, qtd):
        return [Individual(genes=np.random.rand(self.n)) for _ in range(qtd)]

    @abc.abstractmethod
    def calculate_fitness(self, ind, **fitness_args):
        raise NotImplementedError("Method not implemented!")

    def calculate_all_fitness(self, population):
        for ind in population:
            ind.solution = self.decoder(ind)
            ind.fitness = self.calculate_fitness(ind)

    def get_best_ind_from_population(self):
        best_fn = max if self.maximize else min
        return best_fn(chain(*self.population), key=attrgetter("fitness"))

    def create_offsprings(self, elite, non_elite, offs_len):
        offs = []
        for _ in range(offs_len):
            # TODO:What about the seed?
            parent1, parent2 = choice(elite), choice(non_elite)
            new_ind = self.make_crossover(parent1, parent2)
            offs.append(new_ind)
        return offs

    def flip(self):
        return random() < self.get_rho_e()

    def make_crossover(self, parent1, parent2):
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("The genes of individuals must have same size")

        genes = [
            g1 if self.flip() else g2 for g1, g2 in zip(parent1.genes, parent2.genes)
        ]

        return Individual(genes=genes)

    def exchanging_best_chromosomes(self, population):
        t = self.t
        for i in range(self.k):
            if i < self.k - 1:
                population[i + 1][-t:] = population[i][:t]
            else:
                population[0][-t:] = population[-1][:t]

        return population

    def log(self, ind, time_ind):
        self.output[self.generation] = {
            "time": time_ind,
            "solution": ind.solution,
            "fitness": ind.fitness,
            "genes": ind.genes, }

    def run(self):
        get_fitness = attrgetter("fitness")
        start_time = time.time()
        for i in range(self.k):
            self.population[i] = self.create_individuals(self.pop_size)
            self.calculate_all_fitness(self.population[i])

        self.update_best_ind(self.get_best_ind_from_population())
        previous_fitness = self.best_ind.fitness
        any_criterion_met = False

        while not any_criterion_met:
            self.generation += 1
            print("Generation: ", self.generation)
            # print(self.population)

            if (self.k > 1) and (self.generation % self.x_intvl == 0):
                for i in range(self.k):
                    self.population[i].sort(key=get_fitness, reverse=self.maximize)
                self.population = self.exchanging_best_chromosomes(self.population)

            for i in range(self.k):
                self.population[i].sort(key=get_fitness, reverse=self.maximize)

                pop_elite = self.population[i][: self.elite_size]
                pop_non_elite = self.population[i][self.elite_size:]

                mutants = self.create_individuals(self.mutant_size)
                offs_len = self.pop_size - self.elite_size - self.mutant_size
                offsprings = self.create_offsprings(pop_elite, pop_non_elite, offs_len)

                population = mutants + offsprings
                self.calculate_all_fitness(population)
                self.population[i] = pop_elite + population

            if self.maximize:
                [[self.log(self.population[j][i], time.time() - start_time)
                  for i in range(len(self.population[j]))
                  if self.population[j][i].fitness > self.best_ind.fitness]
                 for j in range(len(self.population))]
            else:
                [[self.log(self.population[j][i], time.time() - start_time)
                  for i in range(len(self.population[j]))
                  if self.population[j][i].fitness < self.best_ind.fitness]
                 for j in range(len(self.population))]

            self.update_best_ind(self.get_best_ind_from_population())

            print(self.best_ind)
            if not self.has_improvement(previous_fitness):
                self.generations_no_improvement += 1
            else:
                self.generations_no_improvement = 0
                previous_fitness = self.best_ind.fitness

            any_criterion_met = self.criteria.is_met(self)

        self.runtime = time.time() - start_time
