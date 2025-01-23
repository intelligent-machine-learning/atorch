import gc


class ManualGarbageCollection:
    def __init__(self, gc_freq=300, disable_auto_gc=False):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        if disable_auto_gc:
            self.disable_auto_gc()
            gc.collect()

    def enable_auto_gc(self):
        gc.enable()

    def disable_auto_gc(self):
        gc.disable()

    def run(self, step_count):
        if step_count > 1 and step_count % self.gc_freq == 0:
            gc.collect(1)
