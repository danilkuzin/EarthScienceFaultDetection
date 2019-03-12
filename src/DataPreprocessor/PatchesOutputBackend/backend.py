import numpy as np

class PatchesOutputBackend:
    def save(self, array:np.array, label:int, path: str) -> None:
        pass

    def load(self, path: str) -> np.array:
        pass