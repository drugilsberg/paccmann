"""Initialization for `paccmann.model.model_specifications` submodule."""
from .dnn import dnn_fn
from .rnn import rnn_fn
from .scnn import scnn_fn
from .sa import sa_fn
from .ca import ca_fn
from .mca import mca_fn


# NOTE: model specification factory
MODEL_SPECIFICATION_FACTORY = {
    'dnn': dnn_fn,
    'rnn': rnn_fn,
    'scnn': scnn_fn,
    'sa': sa_fn,
    'ca': ca_fn,
    'mca': mca_fn
}
