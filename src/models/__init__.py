from .food101 import Food101  # export architectures classes
from .food101_0 import Food101_0

# This allows: model_cls = architectures["Food101"]
architectures = {
    "Food101": Food101,
    "Food101_0": Food101_0
}
