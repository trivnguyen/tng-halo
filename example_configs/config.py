
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.training_seed = 10

    config.workdir = 'test'
    config.name = 'debug'
    config.overwrite = True
    config.checkpoint = None
    config.enable_progress_bar = True

    config.data = data = ConfigDict()
    # add your dataset options here

    config.model = model = ConfigDict()
    model.input_size = 2
    model.hidden_sizes = [16, 16]
    model.embed_size = 16
    model.graph_layer = "ChebConv"
    model.graph_layer_args = {"K": 2}
    model.activation_name = "GELU"
    model.layer_norm = True
    model.norm_first = True

    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.weight_decay = 1e-5
    config.scheduler = scheduler = ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.decay_steps = 100_000   # this should not exceed training.num_steps
    scheduler.warmup_steps = 10_000  # about 5-10% of decay_steps
    scheduler.eta_min = 1e-6

    config.training = training = ConfigDict()
    training.batch_size = 64
    training.num_steps = 100_000
    training.patience = 1_000  # early stopping patience
    training.save_top_k = 5  # save the top k models
    training.accelerator = 'cpu'  # 'cpu', 'gpu', 'tpu'

    return config