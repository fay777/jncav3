import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv

from solver.learning.gat_seq2seq_ppo.net import ActorCritic


class ArchitectureAblationTests(unittest.TestCase):

    @staticmethod
    def make_policy(graph_encoder='egat', use_lstm=True):
        return ActorCritic(
            p_net_num_nodes=4,
            p_net_feature_dim=7,
            p_net_edge_feature_dim=3,
            v_net_feature_dim=5,
            embedding_dim=8,
            num_lstm_layers=2,
            num_egat_layers=2,
            num_attention_heads=2,
            graph_encoder=graph_encoder,
            use_lstm=use_lstm,
        )

    @staticmethod
    def make_policy_observation(policy):
        v_net_x = torch.randn(1, 3, 5)
        encoder_outputs = policy.encode({'v_net_x': v_net_x})
        hidden_state = policy.get_last_rnn_state()
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ], dtype=torch.long)
        graph = Data(
            x=torch.randn(4, 7),
            edge_index=edge_index,
            edge_attr=torch.randn(edge_index.shape[1], 3),
        )
        return {
            'p_net': Batch.from_data_list([graph]),
            'p_node_id': torch.tensor([4], dtype=torch.long),
            'hidden_state': hidden_state,
            'encoder_outputs': encoder_outputs.permute(1, 0, 2),
            'mask': None,
        }

    def assert_forward_shapes(self, policy):
        obs = self.make_policy_observation(policy)
        logits = policy.act(obs)
        value = policy.evaluate(obs)
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(policy.get_last_rnn_state().shape), (2, 1, 8))

    def test_full_egat_lstm_forward(self):
        policy = self.make_policy(graph_encoder='egat', use_lstm=True)
        self.assertTrue(any(isinstance(module, GATConv) for module in policy.modules()))
        self.assertTrue(any(isinstance(module, nn.LSTM) for module in policy.modules()))
        self.assert_forward_shapes(policy)

    def test_gcn_ablation_replaces_all_gat_layers(self):
        policy = self.make_policy(graph_encoder='gcn', use_lstm=True)
        self.assertTrue(any(isinstance(module, GCNConv) for module in policy.modules()))
        self.assertFalse(any(isinstance(module, GATConv) for module in policy.modules()))
        self.assert_forward_shapes(policy)

    def test_without_lstm_ablation_removes_recurrent_layers(self):
        policy = self.make_policy(graph_encoder='egat', use_lstm=False)
        self.assertFalse(any(isinstance(module, nn.LSTM) for module in policy.modules()))
        self.assert_forward_shapes(policy)

    def test_explicit_full_flags_preserve_default_state_dict_layout(self):
        default_policy = self.make_policy()
        explicit_policy = self.make_policy(graph_encoder='egat', use_lstm=True)
        self.assertEqual(
            tuple(default_policy.state_dict().keys()),
            tuple(explicit_policy.state_dict().keys()),
        )


if __name__ == '__main__':
    unittest.main()
