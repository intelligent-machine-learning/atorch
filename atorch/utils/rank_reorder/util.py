import json
import os


def get_rank_reorder_dir():
    # Default root directory
    default_root_dir = "/mnt/antsys-aibenchmark-wulanhw-datasets/train_logs/rank_reorder"
    # Get the root directory from environment variable if it exists, otherwise use the default
    root_dir = os.getenv("RANK_REORDER_WORKSPACE_DIR", default_root_dir)
    return root_dir


def get_current_task_dir():
    root_dir = get_rank_reorder_dir()
    job_name = get_aistudio_job_name()
    rank_reorder_dir = os.path.join(root_dir, job_name)
    os.makedirs(rank_reorder_dir, exist_ok=True)
    return rank_reorder_dir


def get_rank_mapping_dir():
    rank_mapping_dir = os.path.join(get_current_task_dir(), "rank_mapping")
    os.makedirs(rank_mapping_dir, exist_ok=True)
    return rank_mapping_dir


def get_rank_mapping_file(org_rank):
    rank_mapping_file = os.path.join(get_rank_mapping_dir(), f"org_rank-{org_rank}.json")
    return rank_mapping_file


def save_rank_mapping(org_rank, reordered_rank):
    rank_mapping_file = get_rank_mapping_file(org_rank)
    if not os.path.exists(rank_mapping_file):
        rank_mapping = {
            "org_rank": org_rank,
            "reordered_rank": reordered_rank,
        }
        with open(rank_mapping_file, "w") as f:
            f.write(json.dumps(rank_mapping))


def get_rank_mapping(org_rank):
    rank_mapping_file = get_rank_mapping_file(org_rank)
    with open(rank_mapping_file, "r") as f:
        rank_mapping = json.load(f)
    return rank_mapping


def get_aistudio_job_name():
    job_name = os.getenv("APP_ID")
    return job_name


def get_current_pod_ip():
    return os.getenv("POD_IP")


def get_current_node_rank():
    return int(os.getenv("RANK"))


def get_k8s_client():
    from kubernetes import client, config

    USER_AGENT = "dlrover/29.0.0/python"
    config_path = os.path.join(get_rank_reorder_dir(), "configs/moe_rank.config")
    if os.path.exists(config_path):
        print(f"Load kube config file with path: {config_path}", flush=True)
        config.load_kube_config(config_file=config_path)
    else:
        try:
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                # We are running inside a k8s cluster
                config.load_incluster_config()
                print("Load the incluster config.")
            else:
                # Use user's kube config
                config.load_kube_config()
                print("Load the kube config file.")
        except Exception as ex:
            print("Failed to load configuration for Kubernetes:\n%s", ex)

    k8s_client = client.CoreV1Api()
    k8s_client.api_client.user_agent = USER_AGENT
    return k8s_client


def print_to_pod_log_file(*args, **kwargs):
    pod_ip = get_current_pod_ip()
    log_file = os.path.join(get_current_task_dir(), f"pod-{pod_ip}.log")
    with open(log_file, "a") as file:
        print(*args, file=file, **kwargs)


def print_to_rank_log_file(*args, **kwargs):
    log_file = os.path.join(get_current_task_dir(), "rank_process.log")
    with open(log_file, "a") as file:
        print(*args, file=file, **kwargs)


def set_pod_node_rank(new_node_rank):
    os.environ["RANK"] = str(new_node_rank)
    os.environ["NODE_RANK"] = str(new_node_rank)
    os.environ["WORKER_RANK"] = str(new_node_rank)
