import abc
import time
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any, Union, List

import torch

from brkga.brkga import Individual
from brkga.single import SingleObjectiveBRKGA


@dataclass
class AsTensorPopulationContainer:
    device: torch.device
    population: Any = None
    pop_fitness: Any = None

    def __post_init__(self):
        if self.pop_fitness is None:
            if self.population is not None:
                self.pop_fitness = torch.full(
                    (self.population.shape[0],), -1.00, dtype=torch.float
                )

    def __getitem__(self, item):
        population = self.population[item]
        fitness = self.pop_fitness[item]
        return AsTensorPopulationContainer(
            population=population, pop_fitness=fitness, device=self.device
        )

    def join(self, other_pop):
        new_pop = torch.cat((self.population, other_pop.population), 0)
        new_fitness = torch.cat((self.pop_fitness, other_pop.pop_fitness), 0)
        return AsTensorPopulationContainer(
            population=new_pop, pop_fitness=new_fitness, device=self.device
        )


class CudaAwareBRKGA(SingleObjectiveBRKGA, ABC):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)

        if not device:
            self._cuda_avaliable = torch.cuda.is_available()
            if not self._cuda_avaliable:
                warnings.warn(
                    "A cuda aware class is used but no GPU device was detected; Using cpu instead"
                )

            device = torch.device("cuda" if self._cuda_avaliable else "cpu")

        self._device = device

        self.population: Union[
            List[AsTensorPopulationContainer], List[None]
        ] = self.k * [None]

    def create_population_as_tensor(self, qtd) -> AsTensorPopulationContainer:
        population = torch.rand(qtd, self.n, device=self._device)
        return AsTensorPopulationContainer(population=population, device=self._device)

    @abc.abstractmethod
    def decoder(self, gene, **kwargs):
        """
            Decoder: Function responsible for transforming the vector of
            random keys into a solution for the specific problem.
            It maps a solution to the problem from the vector.
        """
        raise NotImplementedError("Method not implemented!")

    @abc.abstractmethod
    def calculate_fitness(self, solution, *fitness_args):
        raise NotImplementedError("Method not implemented!")

    def sort_population(self) -> List[AsTensorPopulationContainer]:
        for i in range(self.k):
            argsort = torch.argsort(
                self.population[i].pop_fitness, descending=self.maximize
            )
            self.population[i] = self.population[i][argsort]

        return self.population

    def get_best_ind_from_population(self) -> Individual:
        if self.maximize:
            all_best_index = [torch.argmax(p.pop_fitness) for p in self.population]
            fn_compare = max
        else:
            all_best_index = [torch.argmin(p.pop_fitness) for p in self.population]
            fn_compare = min

        all_best_fitness = [
            self.population[i].pop_fitness[best_idx]
            for i, best_idx in enumerate(all_best_index)
        ]

        zipped = zip(range(self.k), all_best_index, all_best_fitness)

        best_pop, best_idx, best_fitness = fn_compare(zipped, key=lambda p: p[-1])

        genes = self.population[best_pop].population[best_idx, :]
        return Individual(
            genes=genes, fitness=best_fitness, solution=self.decoder(genes)
        )

    def calculate_all_fitness(self, population: AsTensorPopulationContainer):
        for i, gene in enumerate(population.population):
            solution = self.decoder(gene)
            population.pop_fitness[i] = self.calculate_fitness(solution)

    def make_crossover(self, parent1, parent2):
        # offs_len X qtd_genes (floats)
        probs = torch.rand(parent1.shape[0], parent1.shape[-1], device=self._device)
        offs = torch.where(probs <= self.get_rho_e(), parent1, parent2)
        return offs

    def create_offsprings(
            self, elite, non_elite, offs_len
    ) -> AsTensorPopulationContainer:

        assert elite.population.shape[-1] == non_elite.population.shape[-1]

        pop_elite_choices = torch.randint(
            0, elite.population.shape[0], (offs_len,), device=self._device
        )
        pop_non_elite_choices = torch.randint(
            0, non_elite.population.shape[0], (offs_len,), device=self._device
        )

        p1 = elite.population[pop_elite_choices, :]  # parents1
        p2 = non_elite.population[pop_non_elite_choices, :]  # parents2

        offs = self.make_crossover(p1, p2)
        return AsTensorPopulationContainer(population=offs, device=self._device)

    def exchanging_best_chromosomes(
            self, population: List[AsTensorPopulationContainer]
    ) -> List[AsTensorPopulationContainer]:
        t = self.t
        for i in range(self.k):
            if i < self.k - 1:
                population[i + 1].population[-t:] = population[i].population[:t]
                population[i + 1].pop_fitness[-t:] = population[i].pop_fitness[:t]
            else:
                population[0].population[-t:] = population[-1].population[:t]
                population[0].pop_fitness[-t:] = population[-1].pop_fitness[:t]

        return population

    def log(self, ind, time_ind):
        self.output[self.generation] = {
            "time": time_ind,
            "solution": self.best_ind.solution.tolist(),
            "fitness": self.best_ind.fitness.item(),
            "genes": self.best_ind.genes.tolist()
        }

    def get_specific_header_params(self):
        params = super(CudaAwareBRKGA, self).get_specific_header_params()
        params.update({"use_cuda": self._cuda_avaliable, "torch_device": self._device})
        return params

    def run(self):
        start_time = time.time()
        for i in range(self.k):
            self.population[i] = self.create_population_as_tensor(self.pop_size)
            self.calculate_all_fitness(self.population[i])

        self.update_best_ind(self.get_best_ind_from_population())
        previous_fitness = self.best_ind.fitness
        any_criterion_met = False

        while not any_criterion_met:
            self.generation += 1
            print("Generation:", self.generation)

            if (self.k > 1) and (self.generation % self.x_intvl == 0):
                self.population = self.sort_population()
                self.population = self.exchanging_best_chromosomes(self.population)

            for i in range(self.k):
                population = self.population[i]

                elite = population[: self.elite_size]
                non_elite = population[self.elite_size:]
                mutants = self.create_population_as_tensor(self.mutant_size)

                offs_len = self.pop_size - self.elite_size - self.mutant_size
                offsprings = self.create_offsprings(elite, non_elite, offs_len)

                new_pop = mutants.join(offsprings)
                self.calculate_all_fitness(new_pop)
                self.population[i] = elite.join(new_pop)

            if self.maximize:
                [[self.log(self.population[j][i], time.time() - start_time)
                  for i in range(len(self.population[j].population))
                  if self.population[j].pop_fitness[i] > self.best_ind.fitness]
                 for j in range(len(self.population))]
            else:
                [[self.log(self.population[j].population[i], time.time() - start_time)
                  for i in range(len(self.population[j].population))
                  if self.population[j].pop_fitness[i] < self.best_ind.fitness]
                 for j in range(len(self.population))]

            self.update_best_ind(self.get_best_ind_from_population())

            print(self.best_ind)
            if not self.has_improvement(previous_fitness):
                self.generations_no_improvement += 1
            else:
                self.generations_no_improvement = 0
                self.output[self.generation] = {
                    "time": time.time() - start_time,
                    "solution": self.best_ind.solution.tolist(),
                    "genes": self.best_ind.genes.tolist(),
                    "fitness": self.best_ind.fitness.item(),
                }
                previous_fitness = self.best_ind.fitness
            any_criterion_met = self.criteria.is_met(self)
            # print("Sem melhoria:", self.generations_no_improvement)

        self.runtime = time.time() - start_time
