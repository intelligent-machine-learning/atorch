import tempfile
import unittest

import torch

from atorch.utils.parse_memory_pickle import parse_args as parse_memory_pickle_args
from atorch.utils.parse_memory_pickle import parse_memory_pickle_file, print_result
from atorch.utils.parse_trace_json import parse_trace_file, print_profiler_summary


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class TestTools(unittest.TestCase):
    def setUp(self):
        self.store_dir = tempfile.TemporaryDirectory()
        torch.cuda.memory._record_memory_history()

    def tearDown(self):
        self.store_dir.cleanup()
        torch.cuda.memory._record_memory_history(enabled=None)

    def gen_trace_file(self):
        linear = TestModule()
        linear.cuda()
        x = torch.randn(10, 10).cuda()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,  # python stack is empty?
            record_shapes=True,
        ) as prof:
            out = []
            for _ in range(100):
                y = linear(x)
                out.append(y)
        profiler_file = f"{self.store_dir.name}/trace.json"
        prof.export_chrome_trace(profiler_file)
        memory_pickle_file = f"{self.store_dir.name}/memory.pickle"
        torch.cuda.memory._dump_snapshot(memory_pickle_file)
        return profiler_file, memory_pickle_file

    def test_parse_trace_file(self):
        profiler_file, memory_pickle_file = self.gen_trace_file()
        summary, kernel_start_time = parse_trace_file(profiler_file)
        print_profiler_summary([summary], [kernel_start_time])

    def test_parse_memory_pickle_file(self):
        profiler_file, memory_pickle_file = self.gen_trace_file()
        args = parse_memory_pickle_args([memory_pickle_file, "--whitelist", "atorch", "torch"])
        blocklist = args.blocklist or []
        whitelist = args.whitelist or []
        filename_counter, final_counter = parse_memory_pickle_file(memory_pickle_file, whitelist)
        print_result(filename_counter, blocklist, whitelist)
