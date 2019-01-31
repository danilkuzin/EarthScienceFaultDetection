import numpy as np

#todo this needs furhter thinking as it is tightly implemente3d in sampling now and works differently for different backends
class PatchesOutputBackend:
    def save(self, array:np.array, label:int, path: str) -> None:
        pass

    def load(self, path: str) -> np.array:
        pass