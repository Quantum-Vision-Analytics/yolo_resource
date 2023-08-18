import os
import sys
cwd = os.getcwd()
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(cwd))
from Auto_Annotator import Parsing
class Quantum_AA_Arguments:
    kwargs:str
    def __init__(self, kwargs:str):
        self.kwargs = kwargs
        self.parser = Parsing()

    def generate_arguments(self):
        args = self.kwargs.split(" ")

        return self.parser.parse_known_args(args)[0]