import grasp

matrix =np.array([[-400, 200, 200, 200, 40, 75, 200, 16, 30],
                  [200, -400, 200, 40, 200, 65, 16, 200, 26],
                  [200, 200, -400, 75, 65, 200, 30, 26, 200],
                  [200, 40, 75, -400, 200, 200, 200, 24, 45],
                  [40, 200, 65, 200, -400, 200, 24, 200, 39],
                  [75, 65, 200, 200, 200, -400, 45, 39, 200],
                  [200, 16, 30, 200, 24, 45, -400, 200, 200],
                  [16, 200, 26, 24, 200, 39, 200, -400, 200],
                  [30, 26, 200, 45, 39, 200, 200, 200, -400]])

optimal = -940
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
