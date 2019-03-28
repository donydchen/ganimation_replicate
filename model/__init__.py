from .base_model import BaseModel
from .ganimation import GANimationModel



def create_model(opt):
    # specify model name here
    if opt.model == "ganimation":
        instance = GANimationModel()
    else:
        instance = BaseModel()
    instance.initialize(opt)
    instance.setup()
    return instance

