# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

# NOTE since local sgd has each rank async, we need to separate the timers to only
# gpus that are synced (non-data-parallel-group)

from typing import List

import torch
import torch.distributed as dist
from megatron.core.timers import DummyTimer, Timer


class Timers:
    """Class for a group of Timers."""

    def __init__(self, log_level, log_option):
        """Initialize group of timers.

        Args:
            log_level (int): Log level to control what timers are enabled.
            log_option (str): Setting for logging statistics over ranks for all the timers. Allowed: ['max', 'minmax', 'all']. # type: ignore # NOQA
        """
        self._log_level = log_level
        allowed_log_options = set(["max", "minmax", "all"])
        assert log_option in allowed_log_options, "input log option {} is invalid. It must be one of {}".format(
            log_option, allowed_log_options
        )
        self._log_option = log_option
        self._timers = {}
        self._log_levels = {}
        self._dummy_timer = DummyTimer()
        self._max_log_level = 2

    def __call__(self, name, log_level=None):
        """Call timer with name and log level."""
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:
            if log_level is not None:
                assert (
                    log_level == self._log_levels[name]
                ), "input log level {} does not match already existing " "log level {} for {} timer".format(
                    log_level, self._log_levels[name], name
                )
            return self._timers[name]
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:
            log_level = self._max_log_level
        assert log_level <= self._max_log_level, "log level {} is larger than max supported log level {}".format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:
            return self._dummy_timer
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)
        self._log_levels[name] = log_level
        return self._timers[name]

    def _get_elapsed_time_all_ranks(self, names, reset, barrier, process_group: dist.ProcessGroup = None):
        """Returns elapsed times of timers in names.
        Assumptions:
            - All the ranks call this function.
            - `names` are identical on all ranks.
        If the above assumptions are not met, calling this function will
        result in hang.

        Args:
            names (List[str]): list of timer names
            reset (bool): reset the timer after recording the elapsed time
            barrier (bool): if set, do a global barrier before time measurments

        Returns:
            torch.tensor: Tensor of size [world_size, len(names)] with times in float.
        """

        # First make sure all the callers are in sync.
        if barrier:
            torch.distributed.barrier(process_group)

        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros((world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device())
        for i, name in enumerate(names):
            if name in self._timers:
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)

        # See the note above for why we are not using gather.
        torch.distributed._all_gather_base(
            rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1), group=process_group
        )

        return rank_name_to_time

    def _get_global_min_max_time(self, names, reset, barrier, normalizer, process_group: dist.ProcessGroup = None):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier, process_group)
        name_to_min_max_time = {}
        for i, name in enumerate(names):
            rank_to_time = rank_name_to_time[:, i]
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]
            # If the timer exists:
            if rank_to_time.numel() > 0:
                name_to_min_max_time[name] = (
                    rank_to_time.min().item() / normalizer,
                    rank_to_time.max().item() / normalizer,
                )
        return name_to_min_max_time

    def _get_global_min_max_time_string(
        self, names, reset, barrier, normalizer, max_only, process_group: dist.ProcessGroup = None
    ):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer, process_group)
        if not name_to_min_max_time:
            return None
        if max_only:
            output_string = "max time across ranks (ms):"
        else:
            output_string = "(min, max) time across ranks (ms):"
        for name in name_to_min_max_time:
            min_time, max_time = name_to_min_max_time[name]
            if max_only:
                output_string += "\n    {}: {:.2f}".format((name + " ").ljust(48, "."), max_time)
            else:
                output_string += "\n    {}: ({:.2f}, {:.2f})".format((name + " ").ljust(48, "."), min_time, max_time)
        return output_string

    def _get_all_ranks_time_string(self, names, reset, barrier, normalizer, process_group: dist.ProcessGroup = None):
        """Report times across all ranks."""
        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier, process_group)

        output_string = "times across ranks (ms):"
        no_reported_timing = True
        for i, name in enumerate(names):
            not_yet_found = True
            for rank in range(torch.distributed.get_world_size()):
                if rank_name_to_time[rank, i] > 0:
                    no_reported_timing = False
                    if not_yet_found:
                        not_yet_found = False
                        output_string += "\n  {}:".format(name)
                    output_string += "\n     rank {:2d}: {:.2f}".format(rank, rank_name_to_time[rank, i] / normalizer)
        if no_reported_timing:
            return None
        return output_string

    def get_all_timers_string(
        self,
        names: List[str] = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
        process_group: dist.ProcessGroup = None,
    ):
        """Returns the output string with logged timer values according to configured options.

        Args:
            names (List[str]): Names of the timers to log. If None, all registered timers are fetched. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.

        Raises:
            Exception: Raises if log option is invalid.

        Returns:
            str: Formatted string with the timer values.
        """

        if names == None:  # get all registered timers # type: ignore # NOQA
            names = self._timers.keys()

        assert normalizer > 0.0
        if self._log_option in ["max", "minmax"]:
            max_only = False
            if self._log_option == "max":
                max_only = True
            output_string = self._get_global_min_max_time_string(
                names, reset, barrier, normalizer / 1000.0, max_only, process_group
            )
        elif self._log_option == "all":
            output_string = self._get_all_ranks_time_string(names, reset, barrier, normalizer / 1000.0, process_group)
        else:
            raise Exception("unknown timing log option {}".format(self._log_option))
        return output_string

    def log(
        self,
        names: List[str],
        rank: int = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
        process_group: dist.ProcessGroup = None,
    ):
        """logs the timers passed in names to stdout. Example usage is to log average per step value for timer 'foo',
          this function can be called with normalizer factor set to logging interval.

        Args:
            names (List[str]): Names of the timers to log.
            rank (int, optional): logs the timers to a specific rank. If set to None, logs to the last rank. Defaults to None. # type: ignore # NOQA
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.
        """

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier, process_group)
        # If no input rank is provided, log on last rank.
        if rank is None:
            rank = torch.distributed.get_world_size() - 1

        # We should not allow duplicate logging, so stick to the world rank
        if rank == torch.distributed.get_rank() and output_string is not None:
            print(output_string, flush=True)

    def write(
        self,
        names: List[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
        process_group: dist.ProcessGroup = None,
    ):
        """Write timers to a tensorboard writer. Note that we only report maximum time across ranks to tensorboard.

        Args:
            names (List[str]): Names of the timers to log.
            writer (SummaryWriter): Tensorboard SummaryWriter object
            iteration (int): Current iteration.
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer, process_group)
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                writer.add_scalar(name + "-time", max_time, iteration)
