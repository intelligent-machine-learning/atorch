import json
import os
import time

import pandas as pd

from atorch.utils.rank_reorder.util import (
    get_aistudio_job_name,
    get_current_node_rank,
    get_current_pod_ip,
    get_current_task_dir,
    get_k8s_client,
    get_rank_mapping,
    get_rank_mapping_file,
    get_rank_reorder_dir,
    print_to_pod_log_file,
    print_to_rank_log_file,
    save_rank_mapping,
    set_pod_node_rank,
)


def generate_cluster_node_tor_info_from_node_router_csv(csv_file):
    """
    This function reads a CSV file containing node and tor information,
    extracts unique (node_ip, tor_ip) pairs, and returns two dictionaries:
    one mapping node_ip to tor_ip and another mapping tor_ip to a list of node_ips.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    node_ip_name = "b_node_ip"
    tor_ip_name = "a_node_name"

    # Create a set to store unique (node_ip, tor_ip) tuples
    ip_tor_set = set()

    # Iterate through each row of the DataFrame
    for _, row in df.iterrows():
        node_ip = row[node_ip_name]
        tor_ip = row[tor_ip_name]

        # Create a tuple and add it to the set
        ip_tor_set.add((node_ip, tor_ip))

    node_tor_dict = {}
    tor_node_dict = {}
    # Convert the set of tuples to a dictionary
    for node_ip, tor_ip in ip_tor_set:
        node_tor_dict[node_ip] = tor_ip
        if tor_ip not in tor_node_dict:
            tor_node_dict[tor_ip] = []
        tor_node_dict[tor_ip].append(node_ip)

    return node_tor_dict, tor_node_dict


def get_cluster_tor_node_dict():
    cluster_node_tor_router_file = os.path.join(get_rank_reorder_dir(), "csvs", "hyun2-node-network.csv")
    _, cluster_tor_node_dict = generate_cluster_node_tor_info_from_node_router_csv(cluster_node_tor_router_file)
    return cluster_tor_node_dict


def get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict):
    """
    This function assigns ranks to cluster nodes based on their IPs and the static TOR (Top of Rack) node dictionary.
    It first filters the nodes that are part of the cluster, then assigns ranks to nodes that are paired under the
    same TOR, followed by assigning ranks to nodes that are not paired.
    """
    training_node_ip_set = set(training_node_ip_list)
    # to do: add print ip. add print time
    print_to_rank_log_file(f"reorder_node_ranking. training_node_ip_set len: {len(training_node_ip_set)}")
    print_to_rank_log_file(f"training_node_ip_set: {training_node_ip_set}")

    # Get training tor_ips
    training_tor_node_dict = {}
    for tor, n_ips in cluster_tor_node_dict.items():
        now_n_ips = []
        for n_ip in n_ips:
            if n_ip in training_node_ip_set:
                now_n_ips.append(n_ip)

        if len(now_n_ips) > 0:
            training_tor_node_dict[tor] = now_n_ips

    training_ip_ranks_dict = {}
    current_node_rank = 0

    # find paired ips
    paired_count = 0
    for tor, n_ips in training_tor_node_dict.items():
        if len(n_ips) == 2:
            for now_ip in n_ips:
                training_ip_ranks_dict[now_ip] = current_node_rank
                current_node_rank += 1
                paired_count += 1

    print_to_rank_log_file(f"reorder_node_ranking. cluster_tor_node_dict, paired nodes counts: {paired_count}")

    # find not paired ips
    single_count = 0
    for tor, n_ips in training_tor_node_dict.items():
        if len(n_ips) < 2:
            now_ip = n_ips[0]
            training_ip_ranks_dict[now_ip] = current_node_rank
            current_node_rank += 1
            single_count += 1

    print_to_rank_log_file(f"reorder_node_ranking. cluster_tor_node_dict, single nodes counts: {single_count}")
    return training_ip_ranks_dict


def get_training_node_ip_list():
    k8s_client = get_k8s_client()
    job_name = get_aistudio_job_name()
    namespace = "kubemaker"

    label_selector = f"kubemaker={job_name},elasticdl-replica-type=worker"
    pod_list = k8s_client.list_namespaced_pod(
        namespace,
        label_selector=label_selector,
        timeout_seconds=60,
    )
    # to do log

    pod_py_list = [pod for pod in pod_list.items]

    pod_name_list = [pod.metadata.name for pod in pod_list.items]

    print_to_rank_log_file(f"after k8s_client.list_namespaced_pod. pod_name_list len: {len(pod_name_list)}")
    print_to_rank_log_file(f"after k8s_client.list_namespaced_pod. pod_name_list :\n{pod_name_list}")

    # wait pod_ip
    while True:
        pod_list = k8s_client.list_namespaced_pod(
            namespace,
            label_selector=label_selector,
            timeout_seconds=60,
        )
        cluster_node_ip_list = sorted([pod.status.pod_ip for pod in pod_list.items if pod.status.pod_ip])

        print_to_rank_log_file(
            f"after k8s_client.list_namespaced_pod. cluster_node_ip_list len: \n{len(cluster_node_ip_list)}"
        )
        if len(pod_py_list) == len(cluster_node_ip_list):
            print_to_rank_log_file("after k8s_client.list_namespaced_pod. cluster_node_ip_list len equal pod_py_list")
            return cluster_node_ip_list
        else:
            print_to_rank_log_file(
                "after k8s_client.list_namespaced_pod. cluster_node_ip_list len not equal pod_py_list. sleep 3s to retry"  # noqa: E501
            )
            print_to_rank_log_file(
                f"after k8s_client.list_namespaced_pod. cluster_node_ip_list: \n{cluster_node_ip_list}"
            )
            time.sleep(10)


def get_reordered_rank_ip_dict():
    """
    This function returns the rank of the current node based on its IP address.

    The rank reordering logic is as follows:
    1. Retrieve the cluster TOR (Top of Rack) node dictionary.
    2. Get the list of current cluster IPs.
    3. Assign ranks to the nodes based on their IPs and the static TOR node dictionary:
       - Nodes that are paired under the same TOR are assigned ranks first.
       - Nodes that are not paired are assigned ranks next.
    """
    cluster_tor_node_dict = get_cluster_tor_node_dict()
    print_to_rank_log_file(f"reorder_node_ranking. cluster_tor_node_dict len: {len(cluster_tor_node_dict)}")

    # get current cluster ips
    training_node_ip_list = get_training_node_ip_list()
    print(f"reorder_node_ranking. training_node_ip_list len: {len(training_node_ip_list)}", flush=True)

    # get
    training_ip_ranks_dict = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
    print(f"reorder_node_ranking. training_ip_ranks_dict len: {len(training_ip_ranks_dict)}", flush=True)

    # get current node ip
    return training_ip_ranks_dict


def reorder_node_rank():
    # prepare cluster node ip
    current_pod_ip = get_current_pod_ip()

    training_ip_rank_dict_file_path = os.path.join(get_current_task_dir(), "training_ip_rank_dict.json")

    # todo: handle restart situation
    if get_current_node_rank() == 0:
        print_to_rank_log_file(f"{current_pod_ip} prepare to get_reordered_rank_ip_dict")
        training_ip_ranks_dict = get_reordered_rank_ip_dict()
        if os.path.exists(training_ip_rank_dict_file_path):
            print_to_rank_log_file(
                f"{current_pod_ip} found training_ip_rank_dict exists. it means the original rank 0 node failed and "
                "edl launch another node as node 0. no need to "
            )
        else:
            print_to_rank_log_file(f"{current_pod_ip} create training_ip_rank_dict")
            with open(training_ip_rank_dict_file_path, "w") as f:
                f.write(json.dumps(training_ip_ranks_dict))
        print(f"reorder_node_rank-node {current_pod_ip} create training_ip_rank_dict done\n", flush=True)
    else:
        load_success = False
        while True:
            if os.path.exists(training_ip_rank_dict_file_path):
                # 大规模并发读的时候，可能会出问题，所以增加容错
                print_to_pod_log_file(f"{current_pod_ip} found training_ip_rank_dict_file_path. prepare to load")
                attempt = 0
                while attempt < 3:
                    try:
                        with open(training_ip_rank_dict_file_path, "r") as f:
                            training_ip_ranks_dict = json.load(f)
                        print_to_pod_log_file(
                            f"{current_pod_ip} load training_ip_rank succeed. prepare to set load_success"
                        )
                        load_success = True
                        break
                    except OSError as e:
                        attempt += 1
                        print_to_pod_log_file(
                            f"Attempt {attempt}: Failed to read {training_ip_rank_dict_file_path} with error: {e}"
                        )
                        if attempt < 3:
                            time.sleep(3)
                        else:
                            raise e
            else:
                print_to_pod_log_file(f"{current_pod_ip} not found training_ip_rank_dict_file_path. sleep 3s to retry")
                time.sleep(3)
            if load_success:
                print_to_pod_log_file(f"{current_pod_ip} load training_ip_rank succeed")
                break

    print_to_pod_log_file("Here is the training_ip_ranks_dict: \n")
    print_to_pod_log_file(json.dumps(training_ip_ranks_dict, indent=4))

    org_rank = get_current_node_rank()
    rank_mapping_file = get_rank_mapping_file(org_rank)
    if os.path.exists(rank_mapping_file):
        print_to_pod_log_file(f"\nnode with org_rank {org_rank} restarts. prepare to load reordered_rank")
        mapping_dict = get_rank_mapping(org_rank)
        print_to_pod_log_file(
            f"node {current_pod_ip} restart with org_rank {org_rank}. loading mapping_dict: {mapping_dict}"
        )
        reordered_rank = mapping_dict["reordered_rank"]
    else:
        reordered_rank = training_ip_ranks_dict[current_pod_ip]
        print_to_pod_log_file(f"node {current_pod_ip} prepare to call save_rank_mapping")
        save_rank_mapping(org_rank, reordered_rank)
        print_to_pod_log_file(
            f"node {current_pod_ip} first start. org_rank: {org_rank}. reordered_rank: {reordered_rank}"
        )

    set_pod_node_rank(reordered_rank)
    print_to_pod_log_file(f"\npod ip: {current_pod_ip}. org_rank: {org_rank}; reordered_rank: {reordered_rank}")
