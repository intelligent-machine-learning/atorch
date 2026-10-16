from atorch.utils.version import package_version_bigger_than


if package_version_bigger_than("protobuf", "3.20.3"):
    from .protobuf_4_25_3 import coworker_pb2, coworker_pb2_grpc
else:
    from .protobuf_3_20_3 import coworker_pb2, coworker_pb2_grpc


__all__ = ["coworker_pb2", "coworker_pb2_grpc"]
