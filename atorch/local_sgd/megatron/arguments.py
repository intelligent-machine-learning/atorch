# NOTE a safer way to insert megatron args is to provide an extra_args_provider
from argparse import ArgumentParser


def local_sgd_args_provider(parser: ArgumentParser):
    group = parser.add_argument_group(title="local_sgd")
    # local sgd
    group.add_argument("--use-local-sgd", action="store_true", help="This flag enables local sgd training")
    # local sgd training rythm
    group.add_argument(
        "--local-sgd-sync-interval", type=int, default=1, help="Frequency to sync model weights among groups"
    )
    group.add_argument("--local-sgd-warmup-steps", type=int, default=0, help="Warmup step for local sgd")
    group.add_argument(
        "--on-device-local-sgd",
        action="store_true",
        help="This flag disables cpu offload for local sgd, effective only for reducer/outer optim",
    )
    # local sgd skip anomaly
    group.add_argument(
        "--local-sgd-skip-anomaly", action="store_true", help="This flag enables anomaly detection on pseudo grad"
    )
    group.add_argument(
        "--local-sgd-ewma-alpha", type=float, default=0.02, help="Adaptiveness of ewma algorithm for anomaly detection"
    )
    group.add_argument(
        "--local-sgd-ewma-warmup-steps",
        type=int,
        default=120,
        help="Number of model average steps to warmup ewma algorithm",
    )
    group.add_argument(
        "--local-sgd-ewma-threshold", type=int, default=4, help="Threshold for ewma algorithm to determine anomaly"
    )
    # pseudo grad clip/normalization/reduction
    # gta configurations are not specified (set a default?)
    group.add_argument(
        "--local-sgd-pseudo-gradnorm-reduce",
        action="store_true",
        help="Whether to use a softmax reduction on pseudo grad",
    )
    group.add_argument(
        "--local-sgd-weight-softmax-temperature",
        type=float,
        default=None,
        help="To use pseudo gradnorm reduce we can assign a softmax temperature",
    )
    group.add_argument(
        "--local-sgd-clip-pseudo-grad",
        type=float,
        default=None,
        help="If specified, then pseudo grad norm will be clipped",
    )
    group.add_argument(
        "--local-sgd-pseudo-grad-reducer",
        type=str,
        default=None,
        help="The pseudo grad reducer, can be linear or gta, or simply None",
    )
    group.add_argument(
        "--local-sgd-pseudo-grad-normalize",
        action="store_true",
        help="Whether to normalize pseudo grad before reduction",
    )
    # outer optim configs
    group.add_argument("--outer-optimizer", type=str, default="sgd", help="Outer optimizer, None for regular local sgd")
    group.add_argument("--outer-optimizer-lr", type=float, default=0.7, help="Learning rate for outer optimizer")
    group.add_argument("--outer-optimizer-momentum", type=float, default=0.9, help="Momentum for outer optimizer")
    group.add_argument(
        "--outer-optimizer-nesterov", action="store_true", help="This flag enables nesterov update for outer optimizer"
    )

    return parser
