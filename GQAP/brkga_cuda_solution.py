from brkga import brkga, criterion, cuda
import numpy as np


# Instance
file_name = 'data/GQAP_instances/20-15-35.txt'
obj = 1471896
#print(file_name[-12:-4])
in_file = open(file_name)

lines = in_file.read().replace('\n', '').split('$')
n, m = [int(x) for x in lines[0].split()]
lines = lines[1:]
C = np.array([[float(x) for x in matrix_row.split()] for matrix_row in lines[:n*m]])
matrix = np.array([[float(x) for x in matrix_row.split()] for matrix_row in lines[n*m:] if matrix_row!=''])
#print(matrix)

SEED=None
#np.random.seed(SEED)

def fitness(ind):
    return torch.matmul(torch.matmul(torch.t(ind), matrix), ind)

class GQAPCuda(cuda.CudaAwareBRKGA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decoder(self, ind):
        ind.round()
        return ind.to(torch.float64)

    def calculate_fitness(self, ind, **fitness_args):
        return fitness(ind)

# Critérios testados:
GENERATIONS = 10  # Teste para 10 gerações
criteria = criterion.MultipleCriterion([criterion.GenerationsWithNoImprovment(10),])

if __name__ == "__main__":
    #torch.random.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = torch.tensor(matrix, device=device)
    gqap_cuda = GQAPCuda(
        n=matrix.shape[1], 
        pop_size=10,
        mutant_size=0.2,
        elite_size=0.2,
        q_units=[0.3, 0.5],
        initial_seed=SEED,
        rho_e=0.7,
        maximize=False,
        criteria=criteria,  # Critério de parada
        k=3,  # 3 populações independentes
        x_intvl=3,  # Troca de melhores indivíduos entre populações a cada 3 gerações
        t=2,  # Troca de 4 Indivíduos Elite
    )

    print("Device used: ", device)
    with torch.cuda.device(device):
      gqap_cuda.run()

    print(gqap_cuda.best_ind)
    print("\nTime:", gqap_cuda.runtime)
    gqap_cuda.get_results(name_file="CUDA_Instance_"+file_name[-12:-4])
  
