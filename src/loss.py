from typing import Optional


class LossWrapper:
    def __init__(self, model, loss, fitting_method, super_game):
        self.model = model

        self.loss = loss
        
        self.fitting_method = fitting_method

        self.super_game = super_game

    def __call__(self, model_params):
        self.model.s, self.model.c = model_params

        if self.fitting_method == 'ORF':
            projections = self.model.project_mms(self.super_game, True)[0]

            return self.loss(projections)
        
        elif self.fitting_method == 'MM':
            projections = [self.model.project_mm(self.super_game, True)[0]]

            return self.loss(projections)
        
        elif self.fitting_method == 'AE':
            projections = [self.model.project_ae(self.super_game)[0]]

            return self.loss(projections)
        
        elif self.fitting_method == 'MM_AE':
            projected_mm_mean, projected_mm_sd = self.model.project_mm(self.super_game, True)
            projected_ae_mean, projected_ae_sd = self.model.project_ae(self.super_game)
            return self.loss([projected_mm_mean, projected_ae_mean], [projected_mm_sd, projected_ae_sd])


class MSELoss:
    def __init__(self, target: list, scale: bool):
        self.target = target

        self.scale = scale

    def mse(self, projections):
        loss = 0
        for i in range(len(self.target)):
            loss += (projections[i] - self.target[i]) ** 2

        return loss / len(self.target)
    
    # scales each term by its projected std_dev to "dimensionless" z-scores before computing MSE
    def scaled_mse(self, projections, std_devs):
        loss = 0
        for i in range(len(self.target)):
            loss += ((projections[i] - self.target[i]) / std_devs[i]) ** 2

        return loss / len(self.target)  

    def __call__(self, projections: list, std_devs: Optional[list] = None) -> float:
        if self.scale:
            return self.scaled_mse(projections, std_devs)
        
        return self.mse(projections)