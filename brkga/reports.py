from brkga.brkga import BRKGA
import abc
from dataclasses import dataclass
import pandas as pd


@dataclass
class AbstractReport(abc.ABC):
    report_file_name: str
    brkga_obj: BRKGA
    instance: str = None

    def _get_params(self):
        brkga_obj = self.brkga_obj
        return {
            "Instance": self.instance,
            "pop_size": brkga_obj.pop_size,
            "elite_size": brkga_obj.elite_size / brkga_obj.pop_size,
            "mutant_size": brkga_obj.mutant_size / brkga_obj.pop_size,
            "criteria": brkga_obj.criteria,
            "pho_e": brkga_obj.rho_e,
            "k": brkga_obj.k,
            "x_intvl": brkga_obj.x_intvl,
            "t": brkga_obj.t,
            "initial_seed": brkga_obj.initial_seed,
            "epsilon": brkga_obj.epsilon,
            "maximize": brkga_obj.maximize,
            "n": brkga_obj.n,
            **brkga_obj.get_specific_header_params(),
        }

    @abc.abstractmethod
    def make(self, *args, **kwargs):
        raise NotImplementedError()


class PlainTextReport(AbstractReport):
    def make(self, *args, **kwargs):
        n_file = self.report_file_name
        brkga_obj = self.brkga_obj
        header = self._get_params()

        max_param_name = len(max(header.keys(), key=len)) + 1

        with open(f"{n_file}.txt", "w") as file:
            file.write("-------- Params:\n")
            for h, v in header.items():
                file.write(f"{h.ljust(max_param_name)}:{v}\n")

            file.write("\n-------- Best Individual:\n")
            file.write(f">> Genes: {brkga_obj.best_ind.genes}\n")
            file.write(f">> Solution: {brkga_obj.best_ind.solution}\n")
            file.write(f">> Fitness: {brkga_obj.best_ind.fitness}\n")
            file.write(f">> Generation: {brkga_obj.best_ind.generation}\n")

            file.write("\n-------- Data:\n")
            max_param_name = (
                len(max(list(brkga_obj.output.items())[0][1].keys(), key=len)) + 1
            )
            for g, data in brkga_obj.output.items():
                file.write(f"Generation: {g}\n")
                for result, value in data.items():
                    file.write(f">> {result.ljust(max_param_name)}:{value}\n")

            file.close()


@dataclass
class CsvReport(AbstractReport):
    mode: str = "w"
    extra_data: dict = None

    def __post_init__(self):
        self.extra_data = self.extra_data or {}

    def make(self, *args, **kwargs):
        n_file = self.report_file_name
        brkga_obj = self.brkga_obj
        header = self._get_params()
        output = brkga_obj.output
        data = [
            {"generation": gen, **data, **header, **self.extra_data}
            for gen, data in output.items()
        ]
        df = pd.DataFrame(data)
        df.to_csv(n_file, mode=self.mode)
