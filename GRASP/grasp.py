from dataclasses import dataclass
import numpy as np
from typing import Any, Optional
from random import random, randint, choice
from copy import deepcopy
from time import time
import abc
from itertools import chain
from operator import attrgetter
from typing import Union, List

from dwave_qbsolv import QBSolv
from dimod import BinaryQuadraticModel


@dataclass(repr=True)
class Individual:
    solution: Any = None
    fitness: Any = None


@dataclass
class GRASPQUBO(abc.ABC):
  matrix: Any
  alpha: float
  criteria: "Criterion"  # Criterion contains a list containing the BRKGA stopping criterion
  initial_seed: Optional[int] = 0  # choosen seed
  # Allowable deviation for solutions close to optimal
  epsilon: Optional[float] = 1e-5
  maximize: Optional[bool] = True

  def __post_init__(self):
    self.n = self.matrix.shape[1],
    self.best_ind = None
    self.runtime = 0  # Save the running time of the algorithm
    self.generation = 0  # Generation starts at 0
    self.generations_no_improvement = 0
    self.output = {}
      
  def _get_specific_header_params(self):
      return {}

  def get_results(self, name_file):
      n_file = str(name_file)
      header = {
          "Instance": n_file,
          "n": self.n[0],
          "alpha": self.alpha,
          "criteria": self.criteria,
          "initial_seed": self.initial_seed,
          "epsilon": self.epsilon,
          "maximize": self.maximize,
          **self._get_specific_header_params(),
      }
      max_param_name = len(max(header.keys(), key=len)) + 1

      with open(f"{n_file}.txt", "w") as file:
          file.write("-------- Params:\n")
          for h, v in header.items():
              file.write(f"{h.ljust(max_param_name)}:{v}\n")
          file.write("\n-------- Data:\n")
          max_param_name = (len(max(list(self.output.items())[0][1].keys(), key=len)) + 1)
          for g, data in self.output.items():
              file.write(f"Generation: {g}\n")
              for result, value in data.items():
                  file.write(f">> {result.ljust(max_param_name)}:{value}\n")

          file.close()
      print(header)
      print(self.output)
      return {"header": header, "Output": self.output}

    
  def log(self, ind, time_ind):
    self.output[self.generation] = {
                  "time": time_ind,
                  "solution": ind.solution,
                  "fitness": ind.fitness,
                  }


  def semiConstruction(self):
      if random() < self.alpha:
        ind = Individual(solution=np.random.randint(0, 2, size=self.n))
      else:
        ind = Individual(solution=self.pop_qpu())

      ind.fitness = self.fitness(ind.solution)
      return ind

  def localSearch(self, ind, start_time):
    best_neighbor = deepcopy(ind)
    indices = []
    criterion = False
    
    indices = [i for i in range(self.n[0])]

    while(indices):
      for _ in range(self.n[0]):
        i = np.random.choice(indices)
        indices.remove(i)
        neighbor = deepcopy(best_neighbor)
        neighbor.solution[i] = 0 if neighbor.solution[i]==1 else 1
        neighbor.fitness = self.fitness(neighbor.solution)
        
        if self.bestNeighbor(neighbor, best_neighbor):
          self.best(neighbor, start_time)
          best_neighbor = neighbor
          indices = [i for i in range(self.n[0])]
          break
    print("\nbest_neighbor:", best_neighbor)
    return best_neighbor

  def best(self, ind, start_time):
    if not self.best_ind:
      self.best_ind = ind
    
    if self.maximize:
      if ind.fitness > self.best_ind.fitness:
        self.log(ind, time()-start_time)
        return True
    else:
      if ind.fitness < self.best_ind.fitness:
        self.log(ind, time()-start_time)
        return True
    return False

  def bestNeighbor(self, neighbor, best_neighbor):
    if self.maximize:
      if neighbor.fitness > best_neighbor.fitness:
        return True
    else:
      if neighbor.fitness < best_neighbor.fitness:
        return True
    return False

  def log(self, ind, time_ind):
    self.output[self.generation] = {"time": time_ind, "solution": ind.solution, "fitness": ind.fitness}

  def fitness(self, s):
      return np.matmul(np.matmul(np.transpose(s), self.matrix), s)

  def pop_qpu(self):
    Q = BinaryQuadraticModel(self.matrix, "BINARY")
    response = QBSolv().sample(Q)
    samples = list(response.samples())
    return list(samples[0].values())

  def has_improvement(self, previous_fitness: float) -> bool:
    if self.generation == 1:
        return True
    if self.maximize:
        return self.best_ind.fitness > previous_fitness
    return self.best_ind.fitness < previous_fitness

  def run(self):
    start_time = time()

    print("Iteration: 1")
    complete_solutions = {} # Guardar as soluções completas (rodadas com valização cruzada)
    
    s_linha = self.semiConstruction()            # Construction Method
    s_l = self.localSearch(s_linha, start_time)  # Local Search
    
    # Evaluate the solutions
    self.best_ind = s_l
    previous_fitness = self.best_ind.fitness

    any_criterion_met = False

    while not any_criterion_met:
      self.generation += 1
      print("Iteration ", self.generation)
      
      # Semi-greedy construction
      s_linha = self.semiConstruction()                      # Construction Method
      best_neighbor = self.localSearch(s_linha, start_time)  # Local Search
      
      # Evaluate the solutions
      if self.best(best_neighbor, start_time):
        self.best_ind = best_neighbor
      print("s_best:", self.best_ind)
      
      if not self.has_improvement(previous_fitness):
          self.generations_no_improvement += 1
      else:
          self.generations_no_improvement = 0
          previous_fitness = self.best_ind.fitness

      any_criterion_met = self.criteria.is_met(self)  
    self.runtime = time() - start_time
    return self.best_ind  # or optimal[-1]

@dataclass
class Criterion(abc.ABC):
    @abc.abstractmethod
    def is_met(self, grasp_obj: GRASPQUBO):
        raise NotImplementedError()

@dataclass
class MaxGenerations(Criterion):
    max_generations: int

    def is_met(self, grasp_obj: GRASPQUBO):
        return grasp_obj.generation > self.max_generations

@dataclass
class GenerationsWithNoImprovment(Criterion):
    generations_no_improvement: int

    def is_met(self, grasp_obj: GRASPQUBO):
        return grasp_obj.generations_no_improvement == self.generations_no_improvement

@dataclass
class PercentageDeviation(Criterion):
    opt: float
    data: dict = None

    def __post_init__(self):
        self.data = self.data or {}

    def is_met(self, grasp_obj: GRASPQUBO):
        try:
            solh = self.data["eval"](grasp_obj.best_ind.genes)
        except KeyError:
            solh = grasp_obj.best_ind.fitness

        if not grasp_obj.maximize:
            return (solh - self.opt) / np.abs(self.opt) <= grasp_obj.epsilon
        else:
            return (self.opt - solh) / np.abs(self.opt) <= grasp_obj.epsilon

@dataclass
class MultipleCriterion(Criterion):
    criteria: List[Criterion]

    def is_met(self, grasp_obj: GRASPQUBO):
        return any(c.is_met(grasp_obj) for c in self.criteria)

