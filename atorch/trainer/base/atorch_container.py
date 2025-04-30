import sys
from typing import List, Tuple

from dependency_injector import containers, providers
from transformers import TrainerCallback

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.base.ckptloader import CkptLoader
from atorch.trainer.base.ckptsaver import CkptSaver
from atorch.trainer.utils import DistributedType
from atorch.utils.version import is_megatron_version_bigger_than


class AtorchTrainerContainer(containers.DeclarativeContainer):
    __container_instance = None

    # config = providers.Configuration(ini_files=["config.ini"])
    # wiring_config = containers.WiringConfiguration(packages=[__name__, "atorch.trainer"])

    ckpt_saver = providers.Factory(CkptSaver)

    ckpt_loader = providers.Factory(CkptLoader)

    @classmethod
    def create(cls, train_args: AtorchTrainingArgs) -> Tuple["AtorchTrainerContainer", List[TrainerCallback]]:
        if AtorchTrainerContainer.__container_instance is not None:
            logger.error(
                "Not able to create AtorchTrainerContainer more than once, "
                "please try to use override method if you want to config your injection."
            )
            raise ValueError("Trying to create AtorchTrainerContainer more than once.")

        container = cls()

        # TODO: try to "override" container by atorch addon

        if train_args.distributed_type.lower() == DistributedType.MEGATRON.lower():

            # from atorch.trainer.megatron import AtorchMegatronEngine
            # container.train_engine = providers.Factory(AtorchMegatronEngine, training_args=train_args)

            # saver
            if train_args.flash_checkpoint:
                # from atorch.trainer.megatron.utils import megatron_version_support_async_save

                if is_megatron_version_bigger_than("0.6.0", check_equality=False):
                    raise ValueError(
                        "megatron version > 0.6.0 does not support flash_checkpoint args, please set "
                        "flash_checkpoint to False and set megatron args 'async_args' to True to use "
                        "origin async save, or use sync save by only set flash_checkpoint to False."
                    )
                elif not is_megatron_version_bigger_than("0.6.0"):
                    logger.warning(
                        "megatron version < 0.6.0 does not support flash_checkpoint args, will automatically"
                        "switch to sync save mode"
                    )
                    train_args.flash_checkpoint = False
                    # will go to next if condition
                # 0.6 version async save
                else:
                    from atorch.trainer.megatron.megatron_async_save import MegatronAsyncCkptSaver

                    container.ckpt_saver.override(providers.Singleton(MegatronAsyncCkptSaver))

            if not train_args.flash_checkpoint:
                from atorch.trainer.megatron.megatron_ckpt_saver import MegatronOriginSaver

                container.ckpt_saver.override(providers.Singleton(MegatronOriginSaver))

            # loader
            from atorch.trainer.megatron.megatron_ckpt_loader import MegatronOriginSyncLoader

            container.ckpt_loader.override(providers.Singleton(MegatronOriginSyncLoader))

            # don't delete this line, although it seems not in use here. Because container will wire all the imported
            # modules later, it will be easy to import the object to get injected here.
            import atorch.trainer.megatron.megatron_wrapper  # noqa: F401

        # allow atorch addon override the plugins
        if train_args.ant_config:
            try:
                atorch_addon_path = train_args.ant_config.get("atorch_addon_init_path", None)
                if atorch_addon_path is not None:

                    def load_module_from_path(module_name, path):
                        import importlib.util
                        import sys

                        spec = importlib.util.spec_from_file_location(module_name, path)
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        return module

                    ant_addon = load_module_from_path("atorch.ant_addon", atorch_addon_path)  # noqa: F841
                from atorch.ant_addon.atorch_ioccontainer_register import override_container
            except Exception as ex:
                logger.error(
                    "Not able to import atorch.ant_addon.atorch_ioccontainer_register, please check your"
                    "atorch_addon version!"
                )
                raise ex

            override_container(container, train_args)

        if train_args.custom_register_ioc_container is not None:
            train_args.custom_register_ioc_container(container, train_args)

        # wire all loaded modules contains keyword "atorch"
        wire_modules = {module for module in sys.modules.keys() if module.find("atorch") != -1}
        container.wire(modules=wire_modules)

        # setup container
        AtorchTrainerContainer.__container_instance = container

        # collecting trainer callbacks
        trainer_callbacks = []
        for name, provider in container.providers.items():
            if isinstance(provider, providers.Provider):
                try:
                    instance = provider()
                    if isinstance(instance, TrainerCallback):
                        trainer_callbacks.append(instance)
                except Exception as e:
                    logger.warning(f"Unable to get instance with name: {name}: {e}")

        return container, trainer_callbacks
