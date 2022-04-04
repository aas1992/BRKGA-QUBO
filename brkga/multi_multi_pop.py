import abc
import time
import warnings
from abc import ABC
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt

from brkga.brkga import BRKGA, Individual
from brkga.multiobjective_utils import non_dominated_sorting


@dataclass
class MultiObjectivePopulationContainer:
    population: Any = None
    pop_fitness: Any = None
    use_cuda: bool = torch.cuda.is_available()
    device: torch.device = None
    n_objectives: int = 1

    def __post_init__(self):
        if self.use_cuda and not self.device:
            raise ValueError("You must set a device if cuda is enabled")

        if self.pop_fitness is None:
            if self.population is not None:
                shape = (self.population.shape[0], self.n_objectives)
                if self.use_cuda:
                    self.pop_fitness = torch.full(
                        shape, -1.00, dtype=torch.float, device=self.device
                    )
                else:
                    self.pop_fitness = np.full(shape, -1.00, dtype=float)

    def __getitem__(self, item):
        if self.use_cuda and isinstance(item, Iterable):
            item = torch.tensor(item, device=self.device)
            population = self.population.index_select(dim=0, index=item)
            fitness = self.pop_fitness.index_select(dim=0, index=item)
        else:
            population = self.population[item]
            fitness = self.pop_fitness[item]

        return MultiObjectivePopulationContainer(
            population=population,
            pop_fitness=fitness,
            n_objectives=self.n_objectives,
            use_cuda=self.use_cuda,
            device=self.device,
        )

    def join(self, other_pop):
        if self.use_cuda:
            new_pop = torch.cat((self.population, other_pop.population), 0)
            new_fitness = torch.cat((self.pop_fitness, other_pop.pop_fitness), 0)
        else:
            new_pop = np.concatenate((self.population, other_pop.population), axis=0)
            new_fitness = np.concatenate(
                (self.pop_fitness, other_pop.pop_fitness), axis=0
            )

        return MultiObjectivePopulationContainer(
            population=new_pop,
            pop_fitness=new_fitness,
            n_objectives=self.n_objectives,
            use_cuda=self.use_cuda,
            device=self.device,
        )


class MultiObjectiveBRKGA(BRKGA, ABC):
    def __init__(
        self,
        n_objectives,
        weights,
        criterion=None,
        use_cuda=torch.cuda.is_available(),
        device=None,
        maximize=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.criterion = criterion
        if maximize is None:
            self.maximize = tuple([True] * n_objectives)
        else:
            self.maximize = maximize
        self.n_objectives = n_objectives
        # Pymoo deals with minimization so if we want to maximize some functions on the pareto front we must multiply it
        # by -1 and minimize the inverse

        cuda_avaliable = torch.cuda.is_available()
        if use_cuda:
            if not cuda_avaliable:
                warnings.warn(
                    "The param 'use_cuda' is set to True but no GPU device was detected; Using cpu instead"
                )
                device = torch.device("cpu")
            else:
                device = torch.device("cuda" if use_cuda else "cpu")
        elif device:
            raise ValueError("You must not set a device if cuda is unabled")

        self._device = device
        if use_cuda:
            self.sign_for_fronts = [
                torch.tensor(
                    [-1 if self.maximize[i] else 1 for i in range(self.n_objectives)],
                    device=self._device,
                ).repeat(self.pop_size, 1)
                for _ in range(self.k)
            ]
            # self.sign_for_fronts = self.sign_for_fronts.repeat(self.pop_size, 1)
        else:
            self.sign_for_fronts = [
                np.array(
                    [-1 if self.maximize[i] else 1 for i in range(self.n_objectives)]
                )
                for _ in range(self.k)
            ]
            self.sign_for_fronts = [
                np.repeat(s[np.newaxis, :], self.pop_size, axis=0)
                for s in self.sign_for_fronts
            ]

        self.use_cuda = use_cuda
        self.population = [None] * self.k
        self.fronts = [None] * self.k
        self.optimal_front = [None] * self.k
        self.generations_no_improvement = 0
        self.runtime = None

    def check_criterion(self, criterion):
        raise ValueError(
            f"Multi criterion for a {self.__class__.__name__} class is not yet avaliable."
        )

    @abc.abstractmethod
    def create_population(self, qtd) -> MultiObjectivePopulationContainer:
        raise NotImplementedError("Method not implemented!")

    @abc.abstractmethod
    def calculate_fitness(self, ind, *fitness_args):
        raise NotImplementedError("Method not implemented!")

    def pseudo_weights(self, weights, pop_fitness, return_pseudo_weights=False):
        check = tuple([True] * self.n_objectives)

        if self.use_cuda:
            weights = torch.tensor(weights, device=self._device)
            ideal = pop_fitness.min(axis=0).values
            nadir = pop_fitness.max(axis=0).values

            # Normalized distance to the worst solution
            pseudo_weights = (nadir - pop_fitness) / (nadir - ideal)

            # Normalize weights to sum up to one
            pseudo_weights = pseudo_weights / torch.sum(pseudo_weights, axis=1)[:, None]

            # Search for the closest individual having this pseudo weights
            if self.maximize == check:
                I = torch.argmax(torch.sum(torch.abs(pseudo_weights - weights), axis=1))
            else:
                I = torch.argmin(torch.sum(torch.abs(pseudo_weights - weights), axis=1))

        else:
            weights = np.array(weights)
            ideal = pop_fitness.min(axis=0)
            nadir = pop_fitness.max(axis=0)

            # Normalized distance to the worst solution
            pseudo_weights = (nadir - pop_fitness) / (nadir - ideal)

            # Normalize weights to sum up to one
            pseudo_weights = pseudo_weights / np.sum(pseudo_weights, axis=1)[:, None]

            # Search for the closest individual having this pseudo weights
            if self.maximize == check:
                I = np.argmax(np.sum(np.abs(pseudo_weights - weights), axis=1))
            else:
                I = np.argmin(np.sum(np.abs(pseudo_weights - weights), axis=1))

        if return_pseudo_weights:
            return I, pseudo_weights
        else:
            return I

    def _get_best_ind_from_one_population(
        self, idx_population, weights, criterion=None
    ):
        pop_fitness = self.optimal_front[idx_population].pop_fitness

        if criterion is None:  # Execute Pseudo-Weights (default)
            best_index = self.pseudo_weights(
                weights, pop_fitness, return_pseudo_weights=False
            )

        genes = self.optimal_front[idx_population].population[best_index, :]
        fitness = self.optimal_front[idx_population].pop_fitness[best_index]
        print(
            f"\nBest individual chosen from the Pareto Frontier (pop {idx_population}):"
        )
        print("Genes:", genes)
        print("Fitness:", fitness)

        return Individual(genes=genes, fitness=fitness)

    def get_specific_header_params(self):
        params = super(MultiObjectiveBRKGA, self).get_specific_header_params()
        params.update({"use_cuda": self.use_cuda, "torch_device": self._device})
        return params

    def get_best_ind_from_population(self, weights, criterion=None) -> Individual:
        best_fn = max if self.maximize else min

        all_best = [
            self._get_best_ind_from_one_population(i, weights, criterion)
            for i in range(self.k)
        ]

        return best_fn(all_best, key=lambda ind: ind.fitness[0])

    def sort_population(self, pop_idx, order=None) -> MultiObjectivePopulationContainer:
        if order is None:
            if self.n_objectives == 1:
                if self.use_cuda:
                    order = torch.argsort(
                        self.population[pop_idx].pop_fitness,
                        descending=self.maximize[0],
                    )
                else:
                    order = np.argsort(
                        self.population[pop_idx].pop_fitness,
                        descending=self.maximize[0],
                    )
            else:
                raise ValueError(
                    "Tried to sort a population by a multiobjective fitness without calculating the pareto fronts "
                    "first. "
                )
        return self.population[pop_idx][order]

    def calculate_all_fitness(self, population: MultiObjectivePopulationContainer):
        for i, ind in enumerate(population.population):
            population.pop_fitness[i] = self.calculate_fitness(ind)

    def make_crossover(self, parent1, parent2):
        if self.use_cuda:
            # offs_len X qtd_genes (floats)
            probs = torch.rand(parent1.shape[0], parent1.shape[-1], device=self._device)
            offs = torch.where(probs <= self.get_rho_e(), parent1, parent2)
        else:
            probs = np.random.rand(parent1.shape[0], parent1.shape[-1])
            offs = np.where(probs <= self.get_rho_e(), parent1, parent2)
        return offs

    def create_offsprings(
        self, elite, non_elite, offs_len
    ) -> MultiObjectivePopulationContainer:
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
        return MultiObjectivePopulationContainer(
            population=offs,
            n_objectives=self.n_objectives,
            use_cuda=self.use_cuda,
            device=self._device,
        )

    def display(self):
        if self.use_cuda:
            if self.n_objectives == 1:
                raise NotImplementedError("Not implemented yet.")
            if self.n_objectives == 2:
                plt.figure()

                for i in range(self.k):
                    plt.scatter(
                        self.optimal_front[i]
                        .pop_fitness[:, 0]
                        .clone()
                        .detach()
                        .to("cpu"),
                        self.optimal_front[i]
                        .pop_fitness[:, 1]
                        .clone()
                        .detach()
                        .to("cpu"),
                        s=30,
                        facecolors="none",
                        edgecolors="blue",
                    )
                    # Escolha da solução de pareto dado um critério
                    plt.scatter(
                        self.best_ind.fitness[0].clone().detach().to("cpu"),
                        self.best_ind.fitness[1].clone().detach().to("cpu"),
                        marker="x",
                        color="red",
                        s=150,
                    )
                    plt.show()
            elif self.n_objectives == 3:
                x, y, z = (
                    self.optimal_front.pop_fitness[:, 0].clone().detach().to("cpu"),
                    self.optimal_front.pop_fitness[:, 1].clone().detach().to("cpu"),
                    self.optimal_front.pop_fitness[:, 2].clone().detach().to("cpu"),
                )
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                for xs, ys, zs in zip(x, y, z):
                    ax.scatter(xs, ys, zs)
                ax.set_xlabel("F1")
                ax.set_ylabel("F2")
                ax.set_zlabel("F3")
                plt.show()
            else:
                raise ValueError(
                    "Too many objectives to display, there is no implementation for it."
                )
        else:
            if self.n_objectives == 1:
                raise NotImplementedError("Not implemented yet.")
            if self.n_objectives == 2:
                plt.figure()
                plt.scatter(
                    self.optimal_front.pop_fitness[:, 0],
                    self.optimal_front.pop_fitness[:, 1],
                    s=30,
                    facecolors="none",
                    edgecolors="blue",
                )
                # Escolha da solução de pareto dado um critério
                plt.scatter(
                    self.best_ind.fitness[0],
                    self.best_ind.fitness[1],
                    marker="x",
                    color="red",
                    s=150,
                )
                plt.show()
            elif self.n_objectives == 3:
                x, y, z = (
                    self.optimal_front.pop_fitness[:, 0],
                    self.optimal_front.pop_fitness[:, 1],
                    self.optimal_front.pop_fitness[:, 2],
                )
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                for xs, ys, zs in zip(x, y, z):
                    ax.scatter(xs, ys, zs)
                ax.set_xlabel("F1")
                ax.set_ylabel("F2")
                ax.set_zlabel("F3")
                plt.show()
            else:
                raise ValueError(
                    "Too many objectives to display, there is no implementation for it."
                )

    def exchanging_best_chromosomes(self, population):
        t = self.t
        for i in range(self.k):
            if i < self.k - 1:
                population[i + 1].population[-t:] = population[i].population[:t]
                population[i + 1].pop_fitness[-t:] = population[i].pop_fitness[:t]
            else:
                population[0].population[-t:] = population[-1].population[:t]
                population[0].pop_fitness[-t:] = population[-1].pop_fitness[:t]

        return population

    def has_improvement(self, previous_fitness: float) -> bool:
        if self.generation == 1:
            return True
        if self.maximize:
            if self.use_cuda:
                return torch.all(self.best_ind.fitness > previous_fitness).item()
            return np.all(self.best_ind.fitness > previous_fitness)
        if self.use_cuda:
            return torch.all(self.best_ind.fitness < previous_fitness).item()
        return np.all(self.best_ind.fitness < previous_fitness)

    def run(self):
        weights = self.weights
        criterion = self.criterion
        start_time = time.time()
        self.population = [self.create_population(self.pop_size) for _ in range(self.k)]
        for i in range(self.k):
            self.calculate_all_fitness(self.population[i])
            self.fronts[i] = non_dominated_sorting(
                self.sign_for_fronts[i] * self.population[i].pop_fitness,
                use_cuda=self.use_cuda,
                device=self._device,
            )
            self.optimal_front[i] = self.population[i][[i for i in self.fronts[i][0]]]

        self.best_ind = self.get_best_ind_from_population(weights, criterion)
        previous_fitness = self.best_ind.fitness
        any_criterion_met = False

        def cpu_array(array):
            if self.use_cuda:
                return array.cpu()
            return np.array(array)

        while not any_criterion_met:
            print("Generation:", self.generation)
            self.generation += 1
            # now we create a list of sorted individuals according to dominance, but since individuals in the
            # same front are not comparable we shuffle all fronts so that if the elite size is less than an
            # integer number of fronts it will take a random sample of elements from that front

            for i in range(self.k):
                randomized_concatenated_fronts = [
                    j
                    for f in self.fronts[i]
                    for j in np.random.permutation(cpu_array(f))
                ]
                self.population[i] = self.sort_population(
                    i, order=randomized_concatenated_fronts
                )

            if (self.k > 1) and (self.generation % self.x_intvl == 0):
                self.population = self.exchanging_best_chromosomes(self.population)

            for i in range(self.k):
                population = self.population[i]
                elite = population[: self.elite_size]
                non_elite = population[self.elite_size :]
                mutants = self.create_population(self.mutant_size)

                offs_len = self.pop_size - self.elite_size - self.mutant_size
                offsprings = self.create_offsprings(elite, non_elite, offs_len)

                new_pop = mutants.join(offsprings)
                self.calculate_all_fitness(new_pop)
                self.population[i] = elite.join(new_pop)
                self.fronts[i] = non_dominated_sorting(
                    self.sign_for_fronts[i] * population.pop_fitness,
                    use_cuda=self.use_cuda,
                    device=self._device,
                )

                self.optimal_front[i] = population[[j for j in self.fronts[i][0]]]

            self.best_ind = self.get_best_ind_from_population(weights, criterion)

            if not self.has_improvement(previous_fitness):
                self.generations_no_improvement += 1
            else:
                self.generations_no_improvement = 0
                if self.use_cuda:
                    self.output[self.generation] = {
                        "time": time.time() - start_time,
                        "genes": self.best_ind.genes.tolist(),
                        "fitness": self.best_ind.fitness.tolist(),
                    }
                    previous_fitness = self.best_ind.fitness
                else:
                    self.output[self.generation] = {
                        "time": time.time() - start_time,
                        "genes": list(self.best_ind.genes),
                        "fitness": self.best_ind.fitness,
                    }
                    previous_fitness = self.best_ind.fitness

            any_criterion_met = self.criteria.is_met(self)

        self.runtime = time.time() - start_time
