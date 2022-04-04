from brkga import brkga, criterion, default

import numpy as np

#GQAP_source_dir = '/content/drive/MyDrive/Projeto Quantica/GQAP/'
#SEEDS = [268567135, 546551897, 267072247, 315566276, 343409079, 979067383, 230082444, 747425212, 410197446, 482820522, 551527501, 603159793, 604520843, 502226520, 277254279, 528622053, 115299745, 247920999, 907776844, 827579906, 338539033, 278217055, 960001742, 539389713, 767920749, 373501297, 974666969, 271137203, 197040269, 733194481, 613248730, 766892628, 652472791, 517426985, 893400843, 260446978, 722181210, 361839349, 851679693, 460238479, 569004050, 221526167, 374764954, 978231927, 267404990, 876430100, 485080997, 343327687, 313132697, 867740154, 939390418, 941535392, 509073070, 503471256, 474167206, 456131662, 853000884, 269618772, 225650047, 688567763, 549464847, 269712772, 489029680, 577873411, 844001396, 132554764, 494907798, 422908231, 129912083, 636115996, 557711329, 741386022, 394322005, 584134339, 450083816, 952445648, 186156888, 825624942, 619655845, 313800057, 496638874, 934371909, 481445900, 767123499, 264652393, 373045151, 138724525, 965801631, 458927680, 103280333, 831543229, 854635477, 225387878, 135767663, 541718523, 203831246, 732418495, 178082929, 475013861, 400297353]

# Instance
file_name = 'data/GQAP_instances/20-15-35.txt'
optimal = 1471896

#print("file_name: ", file_name)
in_file = open(file_name)
#print(in_file)

lines = in_file.read().replace('\n', '').split('$')
n, m = [int(x) for x in lines[0].split()]
lines = lines[1:]
C = np.array([[float(x) for x in matrix_row.split()] for matrix_row in lines[:n*m]])
matrix = np.array([[float(x) for x in matrix_row.split()] for matrix_row in lines[n*m:] if matrix_row!=''])

print(matrix)
print(matrix.shape)

SEED=None
#np.random.seed(SEED)

# Critérios testados:
criteria = criterion.criterionMultipleCriterion(
            [criterion.PercentageDeviation(opt=optimal),
             # criterion.GenerationsWithNoImprovment(20),
             # criterion.MaxGenerations(500),
            ])

def fitness(ind):
    return np.matmul(np.matmul(np.transpose(np.rint(ind)), matrix), np.rint(ind))

class QAPBrkga(default.DefaultBRKGA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decoder(self, ind, **kwargs):
        np.random.seed(None)
        solution = np.rint(ind)
        return solution

    def calculate_fitness(self, ind, **fitness_args):
        return fitness(ind)


if __name__ == "__main__":
    qaproblem = QAPBrkga(n=matrix.shape[1], q_units=[0.3, 0.5], pop_size=10, mutant_size=0.2, elite_size=0.2, initial_seed=SEED, rho_e=0.7,
                             maximize=False, criteria=criteria, k=3, x_intvl=3, t=2)

    qaproblem.run()

    print(qaproblem.best_ind)
    print("\nTime:", qaproblem.runtime)
    qaproblem.get_results(name_file="GQAP_Default_"+file_name[-12:-4])
