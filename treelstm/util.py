'''Helper functions for running the TreeLSTM model
'''

import numpy
import torch

from torch.utils.data import Dataset, IterableDataset

class TreeDataset(Dataset):

	def __init__(self, data):
		self.data = data
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]


def _label_node_index(node, n=0):
	node['index'] = n
	for child in node['c']:
		n += 1
		_label_node_index(child, n)


def _gather_node_attributes(node, key):
	features = [node[key]]
	for child in node['c']:
		features.extend(_gather_node_attributes(child, key))
	return features


def _gather_adjacency_list(node):
	adjacency_list = []
	for child in node['c']:
		adjacency_list.append([node['index'], child['index']])
		adjacency_list.extend(_gather_adjacency_list(child))

	return adjacency_list


def convert_tree_to_tensors(tree, device=torch.device('cpu')):
	# Label each node with its walk order to match nodes to feature tensor indexes
	# This modifies the original tree as a side effect
	_label_node_index(tree)

	features = _gather_node_attributes(tree, 'f')
	labels = _gather_node_attributes(tree, 'l')

	root_label = [labels[0]]

	# print(root_label)

	adjacency_list = _gather_adjacency_list(tree)

	# print(adjacency_list)

	node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

	root_node = [0]

	return {
		'f': torch.tensor(features, device=device, dtype=torch.float32),
		'l': torch.tensor(labels, device=device, dtype=torch.float32),
		'root_l': torch.tensor(root_label, device=device, dtype=torch.float32),
		'root_n': torch.tensor(root_node, device=device, dtype=torch.int64),
		'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
		'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
		'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
	}

def calculate_evaluation_orders(adjacency_list, tree_size):
    '''Calculates the node_order and edge_order from a tree adjacency_list and the tree_size.

    The TreeLSTM model requires node_order and edge_order to be passed into the model along
    with the node features and adjacency_list.  We pre-calculate these orders as a speed
    optimization.
    '''
    adjacency_list = numpy.array(adjacency_list)

    node_ids = numpy.arange(tree_size, dtype=int)

    node_order = numpy.zeros(tree_size, dtype=int)
    unevaluated_nodes = numpy.ones(tree_size, dtype=bool)

    parent_nodes = adjacency_list[:, 0]
    child_nodes = adjacency_list[:, 1]

    n = 0
    while unevaluated_nodes.any():
        # Find which child nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[child_nodes]

        # Find the parent nodes of unevaluated children
        unready_parents = parent_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of parents with unevaluated child nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_parents)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    edge_order = node_order[parent_nodes]

    return node_order, edge_order


def batch_tree_input(batch):
    '''Combines a batch of tree dictionaries into a single batched dictionary for use by the TreeLSTM model.

    batch - list of dicts with keys ('f', 'node_order', 'edge_order', 'adjacency_list')
    returns a dict with keys ('f', 'node_order', 'edge_order', 'adjacency_list', 'tree_sizes')
    '''
    tree_sizes = [b['f'].shape[0] for b in batch]

    batched_features = torch.cat([b['f'] for b in batch])
    batched_node_order = torch.cat([b['node_order'] for b in batch])

    idx = 0
    root_li = []

    for b in batch:
        root_li.append(idx)
        idx += len(b['node_order'])

    batched_root = torch.tensor(root_li, dtype=torch.int64)

    batched_edge_order = torch.cat([b['edge_order'] for b in batch])

    batched_labels = torch.cat([b['l'] for b in batch])

    batched_root_labels = torch.cat([b['root_l'] for b in batch])

    batched_adjacency_list = []
    offset = 0
    for n, b in zip(tree_sizes, batch):
        batched_adjacency_list.append(b['adjacency_list'] + offset)
        offset += n
    batched_adjacency_list = torch.cat(batched_adjacency_list)

    return {
        'f': batched_features,
        'node_order': batched_node_order,
        'edge_order': batched_edge_order,
        'adjacency_list': batched_adjacency_list,
        'tree_sizes': tree_sizes,
        'root_node': batched_root,
        'root_label': batched_root_labels,
        'l': batched_labels
    }


def unbatch_tree_tensor(tensor, tree_sizes):
    '''Convenience functo to unbatch a batched tree tensor into individual tensors given an array of tree_sizes.

    sum(tree_sizes) must equal the size of tensor's zeroth dimension.
    '''
    return torch.split(tensor, tree_sizes, dim=0)
