import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def construct_label_metrix(labels):
    matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    return matrix

class ContrastiveLoss(nn.Module):
    def __init__(self, criterion, logit_scale):
        super(ContrastiveLoss, self).__init__()
        self.criterion = criterion
        self.logit_scale = logit_scale

    def forward(self, image_features, dna_features, text_features, label, logit_scale=1/0.07):
        feature_list = [image_features, dna_features, text_features]
        feature_list = [item for item in feature_list if item is not None]
        label = construct_label_metrix(label).to(label.device)

        if len(feature_list) < 2:
            raise ValueError("Too less element for calculating the contrastive loss.")

        loss_list = []

        for idx_a, feature_a in enumerate(feature_list):
            for idx_b, feature_b in enumerate(feature_list):
                if idx_a == idx_b:
                    continue
                feature_a = F.normalize(feature_a, p=2, dim=1)
                feature_b = F.normalize(feature_b, p=2, dim=1)

                sim_a_b = self.logit_scale * feature_a @ feature_b.T
                sim_b_a = self.logit_scale * feature_b @ feature_a.T


                loss_a_b = self.criterion(sim_a_b, label)
                loss_b_a = self.criterion(sim_b_a, label)
                loss_list.append(loss_a_b)
                loss_list.append(loss_b_a)
        return sum(loss_list) * 1.0 / len(loss_list)


# Copied from the official OpenCLIP implementation
def gather_features(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        else:
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features[rank] = features
            all_features = torch.cat(gathered_features, dim=0)

    return all_features

# Modified from the official OpenCLIP implementation
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.criterion = criterion

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    def forward(self, image_features, dna_features, text_features, labels, logit_scale, output_dict=False):
        device = image_features.device
        all_image_features = image_features
        all_dna_features = dna_features
        all_text_features = text_features
        all_labels = torch.cat(torch.distributed.nn.all_gather(labels), dim=0)
        all_labels = construct_label_metrix(all_labels).to(device)
        if self.world_size > 1:
            if image_features is not None:
                all_image_features = gather_features(
                    image_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if dna_features is not None:
                all_dna_features = gather_features(
                    dna_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if text_features is not None:
                all_text_features = gather_features(
                    text_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        feature_list = [all_image_features, all_dna_features, all_text_features]
        feature_list = [item for item in feature_list if item is not None]

        if len(feature_list) < 2:
            raise ValueError("Too less element for calculating the contrastive loss.")

        loss_list = []

        for idx_a, feature_a in enumerate(feature_list):
            for idx_b, feature_b in enumerate(feature_list):
                if idx_a == idx_b:
                    continue
                feature_a = F.normalize(feature_a, p=2, dim=1)
                feature_b = F.normalize(feature_b, p=2, dim=1)

                sim_a_b = logit_scale * feature_a @ feature_b.T
                sim_b_a = logit_scale * feature_b @ feature_a.T

                # sim_a_b = feature_a @ feature_b.T
                # sim_b_a = feature_b @ feature_a.T

                loss_a_b = self.criterion(sim_a_b, all_labels)
                loss_b_a = self.criterion(sim_b_a, all_labels)
                loss_list.append(loss_a_b)
                loss_list.append(loss_b_a)

        total_loss = sum(loss_list) * 1.0 / len(loss_list)
        return {"contrastive_loss": total_loss} if output_dict else total_loss