import math
from typing import Callable, Iterator, List, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.base.dataloader import AtorchDataloader
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core import mpu
    from megatron.training import get_args
    from megatron.training.training import build_train_valid_test_data_iterators


_PYTORCH_DATALOADER_KWARGS = {  # default dataloader args
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
    "generator": None,
    "prefetch_factor": 2,
    "persistent_workers": False,
    "pin_memory_device": "",
}


class AtorchMegatronDataloader(AtorchDataloader):
    """
    Dummy dataloader presents model parameters or param groups, this is primarily used to follow conventional training

    Args:
        **dataset_kwargs: Megatron data arguments.
    """

    def __init__(self, **dataset_kwargs):
        # parser = argparse.ArgumentParser()
        # parser = _add_data_args(parser)
        # parser = _add_validation_args(parser)
        # data_args = parser.parse_known_args()
        data_args = get_args()
        self.dataset_args = vars(data_args[0])
        self.dataset_args.update(dataset_kwargs)
        self.dataset_args["megatron_dataset_flag"] = True

    def set_megatron_data_args(self):
        args = self.dataset_args
        for key, value in self.dataset_args.items():
            old_value = getattr(args, key, "")
            if old_value != value:
                print(
                    f"WARNING: MegatronLMDummyDataLoader overriding arguments for "
                    f"{key}:{old_value} with {key}:{value}"
                )
            setattr(args, key, value)

    def get_train_valid_test_datasets_provider(self, megatron_args):
        def train_valid_test_datasets_provider(train_val_test_num_samples):
            """Build train, valid, and test datasets."""
            args = self.dataset_args
            dataset_args = {
                "data_prefix": args.data_path if isinstance(args.data_path, (list, tuple)) else [args.data_path],
                "splits_string": args.split,
                "train_valid_test_num_samples": train_val_test_num_samples,
                "seed": args.seed,
            }
            if args.model_type_name == "bert":
                dataset_args.update(
                    {
                        "max_seq_length": args.seq_length,
                        "binary_head": args.bert_binary_head,
                    }
                )
            elif args.model_type_name == "gpt":
                dataset_args.update(
                    {
                        "max_seq_length": args.seq_length,
                    }
                )
            elif args.model_type_name == "t5":
                dataset_args.update(
                    {
                        "max_seq_length": args.encoder_seq_length,
                        "max_seq_length_dec": args.decoder_seq_length,
                        "dataset_type": "t5",
                    }
                )
            else:
                raise ValueError(f"Unsupported model type: {args.model_type_name}")

            from megatron.legacy.data.dataset_utils import build_train_valid_test_datasets

            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(**dataset_args)
            return train_ds, valid_ds, test_ds

        if megatron_args.custom_megatron_datasets_provider_function is not None:
            return megatron_args.custom_megatron_datasets_provider_function
        try:
            args = self.dataset_args
            # Use '--no-use-pep517 -e' to pip install nvidia's megatron from source
            if args.model_type_name == "bert":
                from pretrain_bert import train_valid_test_datasets_provider  # noqa

                train_valid_test_datasets_provider.is_distributed = True
                return train_valid_test_datasets_provider
            elif args.model_type_name == "gpt":
                from pretrain_gpt import train_valid_test_datasets_provider  # noqa

                train_valid_test_datasets_provider.is_distributed = True
                return train_valid_test_datasets_provider
            elif args.model_type_name == "t5":
                from pretrain_t5 import train_valid_test_datasets_provider  # noqa

                train_valid_test_datasets_provider.is_distributed = True
                return train_valid_test_datasets_provider
        except ImportError:
            pass
        return train_valid_test_datasets_provider

    def build_train_valid_test_data_iterators(self, megatron_args):
        args = self.dataset_args

        train_valid_test_dataset_provider = self.get_train_valid_test_datasets_provider(megatron_args)
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(getattr(args, "model_len", 0)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            (
                train_data_iterator,
                valid_data_iterator,
                test_data_iterator,
            ) = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)

        return train_data_iterator, valid_data_iterator, test_data_iterator


def _prepare_megaton_dataloader(train_args, dataloaders):
    """
    entrance to warp/generate dataloader

    Args:
        train_args:
        dataloaders:

    Returns:

    """
    from atorch.trainer.args import MegatronArgs

    megatron_args: MegatronArgs = train_args.megatron_args()
    micro_batch_size = None

    if not megatron_args.megatron_dataset_flag:
        batch_sizes = [dataloader.batch_size for dataloader in dataloaders if hasattr(dataloader, "batch_size")]
        if len(batch_sizes) == 0:
            raise ValueError(
                "You must specify a training or evaluation dataloader in `accelerate.prepare()` when using Megatron-LM."
            )

        micro_batch_size = min(batch_sizes) if megatron_args.is_train_batch_min else max(batch_sizes)
        if len(batch_sizes) > 1:
            logger.info(
                "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                f"{megatron_args.is_train_batch_min} will decide the `train_batch_size` ({micro_batch_size})."
            )
    else:
        for dataloader in dataloaders:
            if isinstance(dataloader, AtorchMegatronDataloader):
                micro_batch_size = dataloader.dataset_args["micro_batch_size"]
                break
            else:
                raise NotImplementedError("currently only support AtorchMegatronDataloader and subclasses")

    if micro_batch_size is not None:
        # dp_degree = megatron_args.world_size // (
        #     megatron_args.tensor_model_parallel_size * megatron_args.pipeline_model_parallel_size
        # )
        # megatron_args.set_training_args(micro_batch_size, dp_degree)
        pass
    else:
        raise ValueError(
            "When you do not pass the dataloader parameter, the `data_parallel_size`, "
            "`micro_batch_size`, and `global_batch_size` megatron parameters will not be updated."
        )

    batch_data = None

    for dataloader in dataloaders:
        if isinstance(dataloader, DataLoader) and batch_data is None:
            batch_data = next(iter(dataloader))

    torch.distributed.barrier()

    counter = 0
    result = []
    for dataloader in dataloaders:
        if isinstance(dataloader, DataLoader):
            result.append(prepare_data_loader(train_args, dataloader))
            counter += 1
        elif isinstance(dataloader, AtorchMegatronDataloader):
            if counter == 0:
                dataloader.set_megatron_data_args()
                megatron_dataloaders = prepare_data_loader(train_args, dataloader)
            result.append(megatron_dataloaders[counter])  # noqa
            counter += 1

    return tuple(result)


def _prepare_torch_dataloader(dataloader, train_args):  # TODO: add : AtorchTrainingArg, now avoid cycle dep
    logger.info("Preparing dataloader")
    megatron_args = train_args.megatron_args()
    args = get_args()
    if not megatron_args.megatron_dataset_flag:
        micro_batch_size = args.micro_batch_size * megatron_args.num_micro_batches
        kwargs = {k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k]) for k in _PYTORCH_DATALOADER_KWARGS}
        if kwargs["batch_size"] is None:
            if isinstance(kwargs["sampler"], torch.utils.data.BatchSampler):
                kwargs["sampler"].batch_size = micro_batch_size
            else:
                del kwargs["sampler"]
                del kwargs["shuffle"]
                del kwargs["batch_size"]
                kwargs["batch_sampler"].batch_size = micro_batch_size
        else:
            del kwargs["batch_sampler"]
            kwargs["batch_size"] = micro_batch_size

        dataloader = DataLoader(dataloader.dataset, **kwargs)
        # split_batches:
        # Megatron only needs to fetch different data between different dp groups,
        # and does not need to split the data within the dp group.

        from accelerate.data_loader import prepare_data_loader

        return prepare_data_loader(
            dataloader,
            train_args.device,
            num_processes=mpu.get_data_parallel_world_size(),
            process_index=mpu.get_data_parallel_rank(),
            split_batches=False,
            put_on_device=True,
            rng_types=train_args.rng_types.copy(),
            dispatch_batches=train_args.dispatch_batches,
        )


def prepare_data_loader(train_args, dataloader):
    logger.info("Prepare dataloader")
    # megatron_args: MegatronArgs = megatron_args
    megatron_args = train_args.megatron_args()

    if not megatron_args.megatron_dataset_flag:
        return _prepare_torch_dataloader(dataloader, train_args)
    else:
        if megatron_args.consumed_samples is not None:
            (
                megatron_args.consumed_train_samples,
                megatron_args.consumed_valid_samples,
                megatron_args.consumed_test_samples,
            ) = megatron_args.consumed_samples
        else:
            (
                megatron_args.consumed_train_samples,
                megatron_args.consumed_valid_samples,
                megatron_args.consumed_test_samples,
            ) = (0, 0, 0)

        megatron_args.micro_batch_size = megatron_args.micro_batch_size * megatron_args.num_micro_batches

        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = dataloader.build_train_valid_test_data_iterators(megatron_args)

        megatron_args.micro_batch_size = megatron_args.micro_batch_size // megatron_args.num_micro_batches

        train_data_iterator = _handle_megatron_empty_data_iterator(megatron_args, data_iterator=train_data_iterator)
        valid_data_iterator = _handle_megatron_empty_data_iterator(megatron_args, data_iterator=valid_data_iterator)
        test_data_iterator = _handle_megatron_empty_data_iterator(megatron_args, data_iterator=test_data_iterator)

        return train_data_iterator, valid_data_iterator, test_data_iterator


def _handle_megatron_empty_data_iterator(train_args, data_iterator):
    """
    Args:
        data_iterator:
    Returns:

    """

    class DummyMegatronDataloader:
        def __init__(self, len=0) -> None:
            self.len = len

        def __iter__(self):
            return self

        def __next__(self):
            return {}

        def __len__(self):
            return self.len

    is_data_iterator_empty = data_iterator is None
    # is_src_data_iterator_empty = torch.tensor(is_data_iterator_empty, dtype=torch.bool, device=train_args.device)
    is_src_data_iterator_empty = torch.tensor(
        is_data_iterator_empty, dtype=torch.bool, device=torch.cuda.current_device()
    )
    torch.distributed.broadcast(
        is_src_data_iterator_empty,
        0,  # get_tensor_model_parallel_src_rank(),
        # group=get_tensor_model_parallel_group(),
    )
    if not is_data_iterator_empty:
        length = len(data_iterator)
        now_length = torch.tensor(length, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(
            now_length,
            0,  # get_tensor_model_parallel_src_rank(),
            # group=get_tensor_model_parallel_group(),
        )
    else:
        length = 0
        now_length = torch.tensor(length, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(
            now_length,
            0,  # get_tensor_model_parallel_src_rank(),
            # group=get_tensor_model_parallel_group(),
        )
    if not is_src_data_iterator_empty and is_data_iterator_empty:
        return DummyMegatronDataloader(now_length.detach().cpu().item())
    return data_iterator


class MegatronIteratorWrapper:
    """
    MegatronIteratorWrapper is used in pretrain, just one epoch.
    """

    def __init__(
        self,
        data_iterator: Union[Iterator, List[Iterator]],
    ) -> None:
        self.data_iterator = data_iterator

    def __iter__(self):
        return self

    def __next__(self):
        return self.data_iterator


class MegatronDataloaderWrapper:
    """
    MegatronDataloaderWrapper is used in post-training scene, supporting multi-epochs.
    """

    def __init__(
        self,
        dataloader: Union[DataLoader, List[DataLoader]],
        is_post_training,
    ) -> None:
        self.dataloader = dataloader

        # The follow attributes are None if self.dataloader is None.
        self.dataset = None
        self.sampler = None
        self.batch_sampler = None

        dataloader_length = 0
        dataset_length = 0
        batch_size = 0
        drop_last = 1

        def _extract_info_from_dataloader(d):
            assert hasattr(d, "dataset"), "dataloader must has 'dataset' attribute."
            assert hasattr(d, "sampler"), "dataloader must has 'sampler' attribute."
            assert hasattr(d, "batch_sampler"), "dataloader must has 'batch_sampler' attribute."
            self.dataset = d.dataset
            self.sampler = d.sampler
            self.batch_sampler = d.batch_sampler
            assert hasattr(self.batch_sampler, "drop_last")
            assert hasattr(self.batch_sampler, "batch_size") or hasattr(self.batch_sampler, "micro_batch_size")
            _drop_last = int(self.batch_sampler.drop_last)
            _batch_size = (
                getattr(self.batch_sampler, "batch_size", None)
                or getattr(self.batch_sampler, "micro_batch_size", None)
                or 0
            )
            _dataloader_length = len(d)
            _dataset_length = len(self.dataset)

            return _dataloader_length, _dataset_length, _batch_size, _drop_last

        args = get_args()

        if args.virtual_pipeline_model_parallel_size is not None:
            assert isinstance(self.dataloader, list), "VPP requires dataloader to be a list."
            for d in self.dataloader:
                if d is not None:
                    dataloader_length, dataset_length, batch_size, drop_last = _extract_info_from_dataloader(d)
        elif self.dataloader is not None:
            dataloader_length, dataset_length, batch_size, drop_last = _extract_info_from_dataloader(self.dataloader)

        size_info = torch.tensor(
            [dataloader_length, dataset_length, batch_size, drop_last],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        # assert: dataloader on model_parallel_rank 0 must be a real dataloader instance, not be None.
        model_parallel_src_rank = torch.distributed.get_process_group_ranks(group=mpu.get_model_parallel_group())[0]
        torch.distributed.broadcast(size_info, model_parallel_src_rank, group=mpu.get_model_parallel_group())

        # The follow three size info will be set to the value on model_parallel rank 0
        # via broadcast in model_parallel group.
        self.dataloader_length = size_info[0].item()
        self.dataset_length = size_info[1].item()
        self.dataloader_batch_size = size_info[2].item()
        self.drop_last = bool(size_info[3].item())

        # Check batch_size in dataloader
        if not is_post_training:
            assert (
                self.dataloader_batch_size == args.micro_batch_size
            ), "dataloader's batch_size should be micro_batch_size when using Megatron."
        num_microbatches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)

        if self.drop_last:
            self.num_update_steps_per_epoch = self.dataloader_length // num_microbatches
        else:
            self.num_update_steps_per_epoch = math.ceil(self.dataloader_length / num_microbatches)

        if self.dataloader is not None:
            rank = torch.distributed.get_rank()
            if self.dataloader_length == self.dataset_length:
                logger.warning(
                    f"WARNING [Rank {rank}] dataloader and dataset has the same length {self.dataloader_length}, "
                    "unexpected!"
                )
            logger.info(
                f"[Rank {rank}] dataset_length {self.dataset_length} dataloader_length {self.dataloader_length}"
                f" batch_sampler_length {len(self.batch_sampler) if self.batch_sampler is not None else 0}"
                f" batch_size {self.dataloader_batch_size} num_microbatches {num_microbatches}"
                f" num_update_steps_per_epoch {self.num_update_steps_per_epoch} drop_last {self.drop_last}",
            )

        self.skip_batches = 0

        self._reset()

    def _reset(self):
        self._num_yielded = self.skip_batches
        if isinstance(self.dataloader, list):
            self._dataloader_iter = [iter(d) if d is not None else None for d in self.dataloader]
        else:
            self._dataloader_iter = iter(self.dataloader) if self.dataloader is not None else None

    def __iter__(self):
        self._reset()

        logger.info(f" [Rank {torch.distributed.get_rank()}] Reset dataloader, skip first {self._num_yielded} batches.")

        # Skipping first batches only occurs in the first training epoch.
        if self.skip_batches > 0:
            self.skip_batches = 0
        return self

    def __next__(self):
        if self._num_yielded < self.num_update_steps_per_epoch:
            self._num_yielded += 1
            return self._dataloader_iter
        else:
            raise StopIteration

    def __len__(self):
        return self.num_update_steps_per_epoch

    def set_epoch(self, epoch):
        """
        Call self.sampler.set_epoch() if it is not None, only used in finetune scene.
        """

        if self.sampler is not None or self.batch_sampler is not None:
            set_epoch = getattr(self.sampler, "set_epoch", None) or getattr(self.batch_sampler, "set_epoch", None)
            assert set_epoch is not None and isinstance(
                set_epoch, Callable
            ), "dataloader's sampler or batch_sampler should have set_epoch() method when do post-training."
            logger.info(f"Calling set_epoch({epoch})")
            set_epoch(epoch)

    @property
    def num_examples(self):
        """
        Return the num of examples in dataset. self.dataset_length is valid on each rank,
        because it received correct value from model parallel rank 0 via broadcasting.
        """
        return self.dataset_length

    def set_skip_batches(self, skip_batches):
        """
        Set how many batches should be skipped to support resuming training.
        """
        assert skip_batches >= 0, f"skip_batches should be >= 0, but got {skip_batches}."
        if skip_batches >= self.num_update_steps_per_epoch:
            raise ValueError(
                f"skip_batches={skip_batches} >= num_update_steps_per_epoch={self.num_update_steps_per_epoch}, not reasonable!"  # noqa: E501
            )
        self.skip_batches = skip_batches


def wrap_megatron_dataloader(
    data_iterator: Union[Iterator, List[Iterator], DataLoader, List[DataLoader]],
    dataset_type: str,  # ["train", "eval", "test"]
    is_post_training: bool,
):
    """
    Args:
        data_iterator:
    Returns:

    """

    def _check_dataloader_type(dataloader, dataloader_type, pre_info):
        if isinstance(dataloader, list):
            for d in dataloader:
                if d is not None:
                    assert isinstance(
                        d, dataloader_type
                    ), f"In {pre_info}, the dataloader instance must be {dataloader_type} type, but got {type(d)} type."
        elif dataloader is not None:
            assert isinstance(
                dataloader, dataloader_type
            ), f"In {pre_info}, the dataloader instance must be {dataloader_type} type, but got {type(dataloader)} type."  # noqa: E501

    assert dataset_type in ["train", "eval", "test"]

    if is_post_training:
        # torch.util.data.Dataloader type is required for dataloader object in finetune scene.
        _check_dataloader_type(data_iterator, DataLoader, "post-train")

        return MegatronDataloaderWrapper(data_iterator, is_post_training)
    else:
        ####### Compat old code in antllm. To be removed
        rank = torch.distributed.get_rank()
        iterator_type = torch.tensor(  # [None, iterator, dataloader]
            [0],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        if rank == 0:
            if isinstance(data_iterator, list):  # for VPP
                for d in data_iterator:
                    if isinstance(d, Iterator):
                        iterator_type[0] = 1
                        break
                    elif isinstance(d, DataLoader):
                        iterator_type[0] = 2
                        break
            else:
                if isinstance(data_iterator, Iterator):
                    iterator_type[0] = 1
                elif isinstance(data_iterator, DataLoader):
                    iterator_type[0] = 2
        # broadcast 'is_iterator' from rank 0 to other ranks.
        torch.distributed.broadcast(iterator_type, 0)
        iterator_type = iterator_type[0].item()
        ####### Compat old code in antllm. To be removed

        if iterator_type == 0:
            return None
        elif iterator_type == 1:
            _check_dataloader_type(data_iterator, Iterator, "pretrain")
        elif iterator_type == 2:
            _check_dataloader_type(data_iterator, DataLoader, "pretrain")

            if dataset_type == "test":
                return MegatronDataloaderWrapper(data_iterator, is_post_training)
            else:
                if data_iterator is not None:
                    if isinstance(data_iterator, list):  # for VPP
                        data_iterator = [iter(d) if d is not None else None for d in data_iterator]
                    else:
                        if data_iterator is not None:
                            data_iterator = iter(data_iterator)
        else:
            raise ValueError(f"Unexpected iterator_type {iterator_type}, please check the broadcast of iterator_type.")

        return MegatronIteratorWrapper(data_iterator)


class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.
    Should not be used if the original dataloader is a `StatefulDataLoader`.
    """

    def __init__(self, batch_sampler, skip_batches=0):
        self.batch_sampler = batch_sampler
        self.batch_size = batch_sampler.batch_size
        self.drop_last = batch_sampler.drop_last
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    def __len__(self) -> int:
        return len(self.batch_sampler)


def skip_first_batches_for_megatron_dataloader(megatron_dataloader, num_batches=0):
    """
    Creates a `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`. Should not be used if
    the original dataloader is a `StatefulDataLoader`.
    """
    args = get_args()

    assert isinstance(
        megatron_dataloader, MegatronDataloaderWrapper
    ), f"Only support dataloader with MegatronDataloaderWrapper type, but got {type(megatron_dataloader)}."

    dataloader = megatron_dataloader.dataloader

    if args.virtual_pipeline_model_parallel_size:
        """
        assume pp size is 4, vpp size is 3, dataloader on pp stages will be:

        pp_rank
        0:      [DadaLoader(),  None,   None]
        1:      [None,          None,   None]
        2:      [None,          None,   None]
        3:      [None,          None,   DadaLoader()]
        """
        assert isinstance(dataloader, List), "VPP requires dataloader to be a list."

        dataloader_position = -1
        for i, d in enumerate(dataloader):
            if d is not None:
                dataloader_position = i
                dataloader = d
                break
        if dataloader_position == -1:
            dataloader = None

    if dataloader is None:
        if args.virtual_pipeline_model_parallel_size:
            new_dataloader = [None for _ in range(args.virtual_pipeline_model_parallel_size)]
        else:
            new_dataloader = None
    else:
        dataset = dataloader.dataset
        sampler_is_batch_sampler = False
        assert not isinstance(
            dataset, IterableDataset
        ), f"dataset {type(dataset)} is IterableDataset class, not supported resuming training in Megatron. Please contact ATorch developer."  # noqa: E501

        sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
        batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler

        assert args.rampup_batch_size is None, "rampup_batch_size is not supported in post-training."
        num_microbatches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
        skip_microbatches = num_batches * num_microbatches

        new_batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_microbatches)

        # We ignore all of those since they are all dealt with by our new_batch_sampler
        ignore_kwargs = [
            "batch_size",
            "shuffle",
            "sampler",
            "batch_sampler",
            "drop_last",
        ]

        kwargs = {
            k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
            for k in _PYTORCH_DATALOADER_KWARGS
            if k not in ignore_kwargs
        }

        skip_batches_dataloader = DataLoader(dataset, batch_sampler=new_batch_sampler, **kwargs)

        if args.virtual_pipeline_model_parallel_size:
            new_dataloader = [None for _ in range(args.virtual_pipeline_model_parallel_size)]
            new_dataloader[dataloader_position] = skip_batches_dataloader
        else:
            new_dataloader = skip_batches_dataloader

    megatron_dataloader = MegatronDataloaderWrapper(new_dataloader, is_post_training=True)

    megatron_dataloader.set_skip_batches(num_batches)

    return megatron_dataloader
