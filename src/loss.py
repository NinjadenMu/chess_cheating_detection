from typing import Optional


class MSELoss:
    def __init__(self, target: list, scale: bool):
        self.target = target

        self.scale = scale

    def mse(self, projected):
        loss = 0
        for i in range(len(self.target)):
            loss += (projected[i] - self.target[i]) ** 2

        return loss / len(self.target)
    
    # scales each term by its projected std_dev to "dimensionless" z-scores before computing MSE
    def scaled_mse(self, projected, std_devs):
        loss = 0
        for i in range(len(self.target)):
            loss += ((projected[i] - self.target[i]) / std_devs[i]) ** 2

        return loss / len(self.target)  

    def __call__(self, projected: list, std_devs: Optional[list] = None) -> float:
        if self.scale:
            return self.scaled_mse(projected, std_devs)
        
        return self.mse(projected)