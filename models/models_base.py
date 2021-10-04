from models.hypertenet import HyperTeNet


class Models(object):
    def __init__(self, params, device='cuda:0'):
        self.model = HyperTeNet(params, device)

    def get_model(self):
        return self.model
