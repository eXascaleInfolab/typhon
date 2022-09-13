from architecture_loader import ArchitectureLoader
import torch.nn as nn
import torch
import copy


class TyphonModel(nn.Module):
    # Check when loading if it is the same version of this file
    version = 1

    def __init__(self,
        dropout_fe,
        dropouts_dm,
        architecture,
        dsets_names
    ):

        assert isinstance(architecture, str), "Provide an architecture"
        assert dsets_names is not None, "Provide names for the datasets"

        super(TyphonModel, self).__init__()
        self.dropout_fe = dropout_fe
        self.dropouts_dm = dropouts_dm
        self.architecture = architecture
        self.dsets_names = dsets_names

        self.fe = ArchitectureLoader.get_fe(self.architecture, self.dropout_fe)
        self.dms = self.init_dms()

        # Recursively init the weights
        self.fe.apply(self.init_weights)
        self.dms.apply(self.init_weights)
        self.set_dropout(self.dropout_fe, self.dropouts_dm)


    def init_weights(self, module):
        # Skip if module has no weights
        if hasattr(module, 'weight'):
            # Weights cannot be fewer than 2D for Xavier initialization
            if len(module.weight.shape) > 1:
                nn.init.xavier_uniform_(module.weight)


    def init_dms(self):
        dms = nn.ModuleDict({})
        for dset_name in self.dsets_names:
            new_dm = ArchitectureLoader.get_dm(self.architecture, self.dropouts_dm[dset_name])
            dms[dset_name] = new_dm
        return dms


    def forward(self, x, dset_name):
        x = self.fe(x)
        return self.dms[dset_name](x)


    # Sets dropout on model -- IMPORTANT when loading from target state!
    def set_dropout(self, dropout_fe=None, dropouts_dm=None):
        assert (dropout_fe or dropouts_dm), "Need new dropout for DMs and/or FE"
        if dropout_fe:
            self.dropout_fe = dropout_fe
            for mod in self.fe:
                if type(mod) is nn.Dropout:
                    mod.p = self.dropout_fe
        if dropouts_dm:
            self.dropouts_dm = dropouts_dm
            for dset_name, dm in self.dms.items():
                for mod in dm:
                    if type(mod) is nn.Dropout:
                        mod.p = self.dropouts_dm[dset_name]


    # To freeze and unfreeze the feature extractor during hydra
    def freeze_fe(self):
        for name, param in self.named_parameters():
            if 'fe' in name:
                param.requires_grad = False


    def unfreeze_fe(self):
        for name, param in self.named_parameters():
            if 'fe' in name:
                param.requires_grad = True


    # Return separate models, with one FE and one DM each (used for specialization)
    def split_typhon(self):
        models = {}
        for dset_name in self.dsets_names:
            # Use deepcopy to have a new object with new reference
            model = copy.deepcopy(self)
            model.dms = nn.ModuleDict({dset_name: self.dms[dset_name]})
            model.dsets_names = [dset_name]
            models[dset_name] = model
        return models


    def to_state_dict(self):
        variables = {k:v for k, v in vars(self).items() if not k.startswith('_')}
        # Throws an error when loading with double splat operator (and is not needed)
        del variables['training']
        return {
            'fe': self.fe.state_dict(),
            'dms': self.dms.state_dict(),
            'variables': variables
        }


    # Generate new model from state_dict
    @staticmethod
    def from_state_dict(trg_model_state):
        # Check the version of the model
        ret = TyphonModel(**trg_model_state['variables'])
        assert TyphonModel.version == ret.version, "Version not corresponding"
        ret.fe.load_state_dict(trg_model_state['fe'])
        ret.dms.load_state_dict(trg_model_state['dms'])
        ret.set_dropout(ret.dropout_fe, ret.dropouts_dm)
        return ret
