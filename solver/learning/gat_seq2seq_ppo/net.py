import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import scatter


class PaperEGATLayer(nn.Module):
    """GAT layer whose edge features affect attention and node outputs."""
    def __init__(self, input_dim, output_dim, edge_dim, heads=4, concat=True):
        super().__init__()
        self.heads = heads
        self.output_dim = output_dim
        self.concat = concat
        self.gat = GATConv(input_dim, output_dim, heads=heads, concat=concat,
                           edge_dim=edge_dim, add_self_loops=False)
        fused_dim = output_dim * heads if concat else output_dim
        self.edge_projection = nn.Linear(edge_dim, fused_dim, bias=False)

    def forward(self, x, edge_index, edge_attr):
        node_features = self.gat(x, edge_index, edge_attr)
        edge_features = self.edge_projection(edge_attr)
        edge_aggregate = scatter(edge_features, edge_index[1], dim=0,
                                 dim_size=x.size(0), reduce='mean')
        return node_features + edge_aggregate


class PaperEGATEncoder(nn.Module):
    """Two-layer edge-featured GAT with four attention heads.

    The first layer concatenates the head outputs; the second averages them,
    retaining a 64-dimensional representation for the policy network.
    """
    def __init__(self, input_dim, output_dim, edge_dim, hidden_dim=64, heads=4):
        super().__init__()
        self.layer1 = PaperEGATLayer(input_dim, hidden_dim, edge_dim,
                                     heads=heads, concat=True)
        self.layer2 = PaperEGATLayer(hidden_dim * heads, output_dim, edge_dim,
                                     heads=heads, concat=False)

    def forward(self, graph):
        x = F.elu(self.layer1(graph.x, graph.edge_index, graph.edge_attr))
        return self.layer2(x, graph.edge_index, graph.edge_attr)


class PaperGCNEncoder(nn.Module):
    """Two-layer topology-only GCN used for the EGAT architecture ablation.

    It preserves the EGAT encoder's depth and output dimension while removing
    multi-head attention and edge-attribute conditioning.  All downstream
    policy, reward, masking, and optimization logic therefore remains fixed.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)

    def forward(self, graph):
        x = F.elu(self.layer1(graph.x, graph.edge_index))
        return self.layer2(x, graph.edge_index)


class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_feature_dim,
                 v_net_feature_dim, embedding_dim=64, num_lstm_layers=2,
                 num_egat_layers=2, num_attention_heads=4,
                 graph_encoder='egat', use_lstm=True):
        super(ActorCritic, self).__init__()
        # self.encoder = Encoder(v_net_feature_dim, embedding_dim=embedding_dim)
        self.emb = nn.Linear(v_net_feature_dim, embedding_dim)
        self.use_lstm = bool(use_lstm)
        self.num_lstm_layers = num_lstm_layers
        if self.use_lstm:
            self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim,
                                   num_layers=num_lstm_layers, bidirectional=False)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
            )
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, p_net_edge_feature_dim,
                           v_net_feature_dim, embedding_dim, num_lstm_layers, num_egat_layers,
                           num_attention_heads, graph_encoder, self.use_lstm)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, p_net_edge_feature_dim,
                             v_net_feature_dim, embedding_dim, num_lstm_layers, num_egat_layers,
                             num_attention_heads, graph_encoder, self.use_lstm)
        self._last_hidden_state = None
        self._last_hidden_cell = None
        self.v_net_feature_dim = v_net_feature_dim
        self.embedding_dim = embedding_dim

    def encode(self, obs):
        x = obs['v_net_x']
        x = x.permute(1, 0, 2)
        x = F.relu(self.emb(x))
        if self.use_lstm:
            outputs, (hidden_state, cell) = self.encoder(x)
        else:
            # Position-wise transformation followed by permutation-invariant
            # mean pooling removes recurrent order modelling while retaining
            # the tensor interface required by the same attention policy.
            outputs = self.encoder(x)
            pooled = outputs.mean(dim=0, keepdim=True)
            hidden_state = pooled.repeat(self.num_lstm_layers, 1, 1)
        self._last_hidden_state = hidden_state
        self._last_hidden_cell = torch.zeros_like(hidden_state)
        return outputs

    def act(self, obs):
        logits, outputs, hidden_state = self.actor(obs)
        self._last_hidden_state = hidden_state
        # self._last_hidden_cell = hidden_cell
        return logits

    def evaluate(self, obs):
        value = self.critic(obs)
        return value

    def get_last_rnn_state(self):
        return self._last_hidden_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state

class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_feature_dim,
                 v_net_feature_dim, embedding_dim=64, num_lstm_layers=2, num_egat_layers=2,
                 num_attention_heads=4, graph_encoder='egat', use_lstm=True):
        super(Actor, self).__init__()
        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, p_net_edge_feature_dim,
                               embedding_dim=embedding_dim,
                               num_lstm_layers=num_lstm_layers,
                               num_egat_layers=num_egat_layers,
                               num_attention_heads=num_attention_heads,
                               graph_encoder=graph_encoder,
                               use_lstm=use_lstm)

    def forward(self, obs):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        return logits, outputs, hidden_state

class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_feature_dim,
                 v_net_feature_dim, embedding_dim=64, num_lstm_layers=2, num_egat_layers=2,
                 num_attention_heads=4, graph_encoder='egat', use_lstm=True):
        super(Critic, self).__init__()
        self.decoder = Decoder(p_net_num_nodes, p_net_feature_dim, p_net_edge_feature_dim,
                               embedding_dim=embedding_dim,
                               num_lstm_layers=num_lstm_layers,
                               num_egat_layers=num_egat_layers,
                               num_attention_heads=num_attention_heads,
                               graph_encoder=graph_encoder,
                               use_lstm=use_lstm)

    def forward(self, obs):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        value = torch.mean(logits, dim=-1, keepdim=True)
        return value

class Encoder(nn.Module):

    def __init__(self, v_net_feature_dim, embedding_dim=64):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(v_net_feature_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        embeddings = F.relu(self.emb(x))
        # outputs, hidden_state = self.gru(embeddings)
        outputs, (hidden_state, hidden_cell) = self.lstm(embeddings)
        return outputs, hidden_state
    

class Decoder(nn.Module):
    
    def __init__(self, p_net_num_nodes, feature_dim, p_net_edge_feature_dim,
                 embedding_dim=64, num_lstm_layers=2, num_egat_layers=2,
                 num_attention_heads=4, graph_encoder='egat', use_lstm=True):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(p_net_num_nodes + 1, embedding_dim)
        self.att = Attention(embedding_dim)
        if num_egat_layers != 2:
            raise ValueError('The controlled graph-encoder ablation requires two GNN layers.')
        if graph_encoder == 'egat':
            self.gat = PaperEGATEncoder(feature_dim, embedding_dim,
                                        edge_dim=p_net_edge_feature_dim,
                                        hidden_dim=embedding_dim,
                                        heads=num_attention_heads)
        elif graph_encoder == 'gcn':
            self.gat = PaperGCNEncoder(feature_dim, embedding_dim,
                                       hidden_dim=embedding_dim)
        else:
            raise ValueError(f'Unsupported graph_encoder: {graph_encoder}')
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )

        self.use_lstm = bool(use_lstm)
        self.num_lstm_layers = num_lstm_layers
        if self.use_lstm:
            self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=num_lstm_layers)
        else:
            self.lstm = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
            )

    def forward(self, obs):
        """
        obs = {'p_net': obs_p_net, 'p_node_id': obs_p_node_id,
                'hidden_state': obs_hidden_state, 'encoder_outputs': obs_encoder_outputs,
                'mask': None}
        """
        batch_p_net = obs['p_net']    # batch_p_net : DataBatch(x=[100, 5], edge_index=[2, 800], edge_attr=[800, 3], batch=[100], ptr=[2])
        # PyTorch LSTM state: (layers, batch, hidden).  Retain every layer;
        # only the final encoder layer is used to form the attention query.
        hidden_state = obs['hidden_state']

        p_node_embeddings = self.gat(batch_p_net)
        p_node_embeddings = p_node_embeddings.reshape(batch_p_net.num_graphs, -1, p_node_embeddings.shape[-1])

        p_node_id = obs['p_node_id']
        encoder_outputs = obs['encoder_outputs']
        mask = obs['mask']
        p_node_emb = self.emb(p_node_id).unsqueeze(0)  # torch.Size([1, 1, 128])

        context, attention = self.att(hidden_state[-1:], encoder_outputs, mask)

        decoder_input = p_node_emb + context
        if self.use_lstm:
            outputs, (hidden_state, hidden_cell) = self.lstm(
                decoder_input, (hidden_state, torch.zeros_like(hidden_state)))
        else:
            outputs = self.lstm(decoder_input)
            hidden_state = outputs.repeat(self.num_lstm_layers, 1, 1)

        p_node_embeddings = p_node_embeddings + hidden_state[-1].unsqueeze(1)
        logits = self.mlp(p_node_embeddings)

        return logits, outputs, hidden_state
    

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        # 加性注意力机制

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim * num_directions)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.transpose(0, 1).repeat(1, seq_len, 1)  # shape: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))  # shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # shape: (batch_size, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # shape: (batch_size, 1, hidden_dim * num_directions)
        context = context.permute(1, 0, 2)
        return context, attn_weights
