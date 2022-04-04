from GRASP import grasp
import numpy as np
drive.mount('/content/drive')

# Instance
file_name = 'data/GQAP_instances/20-15-35.txt'
optimal = 1471896
in_file = open(file_name)

lines = in_file.read().replace('\n', '').split('$')
n, m = [int(x) for x in lines[0].split()]
lines = lines[1:]
matrix = np.array([[float(x) for x in matrix_row.split()] for matrix_row in lines[n*m:] if matrix_row!=''])
print(matrix)

criteria = grasp.MultipleCriterion(
            [
             #grasp.PercentageDeviation(opt=optimal),
            grasp.GenerationsWithNoImprovment(20),
            grasp.MaxGenerations(500),
            ])

qap = grasp.GRASPQUBO(matrix=matrix,
                alpha=0.7,
                maximize=False,
                criteria=criteria,
                )

qap.run()
print(qap.best_ind)
print("\nTime:", qap.runtime)
qap.get_results(name_file="grasp_gqap_instance")
