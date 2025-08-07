import os
import unittest

from atorch.utils.rank_reorder.reorder import get_training_node_ranks_from_ips


class TestGetTrainingNodeRanksFromIps(unittest.TestCase):
    def setUp(self):
        os.environ["RANK_REORDER_WORKSPACE_DIR"] = "/tmp/"
        os.environ["APP_ID"] = "test_reorder"

    def tearDown(self) -> None:
        super().tearDown()
        os.environ.pop("RANK_REORDER_WORKSPACE_DIR", None)
        os.environ.pop("APP_ID", None)

    def test_empty_input(self):
        result = get_training_node_ranks_from_ips([], {})
        self.assertEqual(result, {})

    def test_all_nodes_paired(self):
        training_node_ip_list = ["node1", "node2", "node3", "node4"]
        cluster_tor_node_dict = {"tor1": ["node1", "node2"], "tor2": ["node3", "node4"]}
        result = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
        expected_result = {"node1": 0, "node2": 1, "node3": 2, "node4": 3}
        self.assertEqual(result, expected_result)

    def test_partially_paired_nodes(self):
        training_node_ip_list = ["node1", "node2", "node3"]
        cluster_tor_node_dict = {"tor1": ["node1"], "tor2": ["node2", "node3"]}
        result = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
        expected_result = {"node1": 2, "node2": 0, "node3": 1}
        self.assertEqual(result, expected_result)

    def test_three_nodes(self):
        training_node_ip_list = ["node1", "node2", "node3", "node4"]
        cluster_tor_node_dict = {"tor1": ["node1"], "tor2": ["node2", "node3", "node4"]}
        result = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
        expected_result = {"node1": 0}
        self.assertEqual(result, expected_result)

    def test_three_nodes_two_in_list(self):
        training_node_ip_list = ["node1", "node2", "node3"]
        cluster_tor_node_dict = {"tor1": ["node1"], "tor2": ["node2", "node3", "node4"]}
        result = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
        expected_result = {"node1": 2, "node2": 0, "node3": 1}
        self.assertEqual(result, expected_result)

    def test_empty_tor_dict(self):
        training_node_ip_list = ["node1", "node2"]
        cluster_tor_node_dict = {}
        result = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
        self.assertEqual(result, {})

    def test_multiple_tors(self):
        training_node_ip_list = ["node1", "node2", "node3", "node4"]
        cluster_tor_node_dict = {
            "tor1": ["node1"],
            "tor2": ["node2"],
            "tor3": ["node3"],
            "tor4": ["node4"],
        }
        result = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
        expected_result = {"node1": 0, "node2": 1, "node3": 2, "node4": 3}
        self.assertEqual(result, expected_result)

    def test_nodes_not_in_cluster(self):
        training_node_ip_list = ["node2", "node3"]
        cluster_tor_node_dict = {"tor1": ["node1"]}
        result = get_training_node_ranks_from_ips(training_node_ip_list, cluster_tor_node_dict)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
