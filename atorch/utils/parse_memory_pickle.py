from __future__ import absolute_import, unicode_literals

import glob
import pickle
from argparse import ArgumentParser
from collections import Counter

from tqdm import tqdm


def filename_is_allow(filename, whitelist):
    if not filename.endswith(".py"):
        return False
    # return True
    for whiteitem in whitelist:
        if whiteitem in filename:
            return True
    return False


def parse_memory_pickle_file(pickle_file, blocklist, whitelist):
    with open(pickle_file, "rb") as f:
        pickle_obj = pickle.load(f)
    filename_counter = Counter()
    max_memory_filename_counter = Counter()
    addr2filename = {}
    total_memory = 0
    max_memory = 0
    for device_trace in pickle_obj["device_traces"]:

        for malloc_record in tqdm(device_trace):
            # # {'alloc', 'free_completed', 'free_requested', 'segment_map', 'segment_unmap'}
            if malloc_record["action"] == "alloc":
                total_memory += malloc_record["size"]
                frames = malloc_record["frames"]
                frame_filenames = []
                # only record the first file in whitelist
                has_record = False
                for frame in frames:
                    if filename_is_allow(frame["filename"], whitelist):
                        filename_lineno = frame["filename"] + ":" + str(frame["line"])
                        if not has_record:
                            filename_counter[filename_lineno] += malloc_record["size"]
                            addr2filename[malloc_record["addr"]] = filename_lineno
                            has_record = True

                        frame_filenames.append(filename_lineno)

            elif malloc_record["action"] == "free_completed":
                total_memory -= malloc_record["size"]
                frames = malloc_record["frames"]
                frame_filenames = []
                has_record = False
                if malloc_record["addr"] in addr2filename:
                    filename_lineno = addr2filename.pop(malloc_record["addr"])
                    filename_counter[filename_lineno] -= malloc_record["size"]
                    has_record = True
            elif malloc_record["action"] == "free_requested":
                pass
                # total_memory -= malloc_record["size"]
            elif malloc_record["action"] == "segment_map":
                pass
                # total_memory += malloc_record["size"]
            elif malloc_record["action"] == "segment_unmap":
                pass
                # total_memory -= malloc_record["size"]
            if total_memory > max_memory:
                max_memory = total_memory
                for name, memory_alloc in filename_counter.items():
                    max_memory_filename_counter[name] = memory_alloc
    print(f"{pickle_file} max_memory: {max_memory/1024/1024:.2f}MB")
    print(f"final total_memory {total_memory/1024/1024:.2f}MB")

    return max_memory_filename_counter, filename_counter


def print_result(filename_counter, blocklist, limit=20):
    MB = 1024 * 1024
    print("=" * 100)
    show_count = 0
    for filename, memory_alloc in filename_counter.most_common():
        skip = False
        for blockname in blocklist:
            if filename.startswith(blockname):
                skip = True
                break
        if skip:
            continue
        show_count += 1
        print(filename, f"alloc{memory_alloc/MB:.2f}MB")
        if show_count > limit:
            break


def parse_args(argv=None):
    parser = ArgumentParser(usage="""python parse_memory_pickle.py memory_pickle_file""")

    parser.add_argument("pickle_files", nargs="*")

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--blocklist", nargs="*")
    parser.add_argument("--whitelist", nargs="*", default=["torch", "atorch", "megatron", "transformer_engine"])
    parser.add_argument("--show_count", type=int, default=20)

    args = parser.parse_args(argv)
    return args


def main():
    args = parse_args()
    # TODO: multiprocess
    if len(args.pickle_files) == 1 and "*" in args.pickle_files[0]:
        filenames = glob.glob(args.pickle_files[0])
    else:
        filenames = args.pickle_files

    blocklist = args.blocklist or []
    whitelist = args.whitelist or []
    for filename in filenames:
        filename_counter, final_counter = parse_memory_pickle_file(filename, whitelist)
        print_result(filename_counter, blocklist, whitelist)


if __name__ == "__main__":
    main()
