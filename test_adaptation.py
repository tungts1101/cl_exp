import torch
import timm
from torch import optim, nn
from petl import vision_transformer_ssf
from copy import deepcopy


# configure model
def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "ssf_" in name:
            param.requires_grad = True
    return model

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"

def collect_params(model):
    params = []
    names = []
    for name, param in model.named_parameters():
        if "ssf_" in name:
            params.append(param)
            names.append(name)
    return params, names


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, mask=None):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, mask)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, mask=None):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    entropy = softmax_entropy(outputs)
    # print(f"Entropy: {entropy}")
    # adapt
    loss = entropy.mean() if mask == None else entropy[mask].mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


if __name__ == '__main__':
    # model = TODO_model()

    # model = tent.configure_model(model)
    # params, param_names = tent.collect_params(model)
    # optimizer = TODO_optimizer(params, lr=1e-3)
    # tented_model = tent.Tent(model, optimizer)

    # outputs = tented_model(inputs)  # now it infers and adapts!

    backbone = timm.create_model('vit_base_patch16_224_ssf', pretrained=True, num_classes=0).cuda()
    backbone = configure_model(backbone)
    check_model(backbone)
    params, param_names = collect_params(backbone)
    print(param_names)

    optimizer = optim.Adam(params, lr=1e-2)
    tented_model = Tent(backbone, optimizer)

    inputs = torch.randn(64, 3, 224, 224).cuda()
    outputs = tented_model(inputs)  # now it infers and adapts!
    print(outputs.shape)