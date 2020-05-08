import torch

from treelstm import TreeLSTM, calculate_evaluation_orders, batch_tree_input, TreeDataset, convert_tree_to_tensors

from torch.utils.data import Dataset, IterableDataset, DataLoader

import os
import codecs
from sklearn.metrics import f1_score
import random
import numpy as np

seed_val = 12

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


tree_path = '../../Dataset/Parsed-Trees/'

test_set = ['charliehebdo.txt']

from random import shuffle

IN_FEATURES = 40
OUT_FEATURES = 2
NUM_ITERATIONS = 10
BATCH_SIZE = 50
HIDDEN_UNITS = 128
LEARNING_RATE = 0.001

files = os.listdir(tree_path)


for test_file in test_set:
	print('Training Set:', set(files) - {test_file})

	test_trees = []
	train_trees = []

	for filename in files:
		input_file = codecs.open(tree_path + filename, 'r', 'utf-8')

		tree_li = []
		pos_trees = []
		neg_trees = []

		for row in input_file:
			s = row.strip().split('\t')

			tweet_id = s[0]
			curr_tree = eval(s[1])

			try:
				curr_tensor, curr_label = convert_tree_to_tensors(curr_tree)
			except:
				continue

			curr_tensor['tweet_id'] = tweet_id

			if curr_label == 1:
				pos_trees.append(curr_tensor)
			else:
				neg_trees.append(curr_tensor)

		input_file.close()


		if filename == test_file:
			tree_li = pos_trees + neg_trees
			test_trees = tree_li
	
		else:			
			tree_li = pos_trees + neg_trees

			shuffle(tree_li)

			train_trees += tree_li
	
	model = TreeLSTM(IN_FEATURES, OUT_FEATURES, HIDDEN_UNITS).train()

	loss_function = torch.nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(model.parameters() , lr = LEARNING_RATE)

	for i in range(NUM_ITERATIONS):
		total_loss = 0

		optimizer.zero_grad()

		curr_tree_dataset = TreeDataset(train_trees)

		train_data_generator = DataLoader(
			curr_tree_dataset,
			collate_fn=batch_tree_input,
			batch_size=BATCH_SIZE,
			shuffle = True
		)

		for tree_batch in train_data_generator:
			try:
				h, h_root, c = model(
					tree_batch['f'],
					tree_batch['node_order'],
					tree_batch['adjacency_list'],
					tree_batch['edge_order'],
					tree_batch['root_node'],
					tree_batch['root_label']
					)
			except:
				continue
			
			labels = tree_batch['l']
			root_labels = tree_batch['root_label']

			loss = loss_function(h_root, root_labels)


			loss.backward()

			optimizer.step()

			total_loss += loss

		print(f'Iteration {i+1} Loss: {total_loss}')
	
	print('Training Complete')

	print('Now Testing:', test_file)

	acc = 0
	total = 0

	pred_label_li = []
	true_label_li = []

	for test in test_trees:
		try:
			h_test, h_test_root, c = model(
				test['f'],
				test['node_order'],
				test['adjacency_list'],
				test['edge_order'],
				test['root_n'],
				test['root_l']
			)
		except:
			continue

		pred_v, pred_label = torch.max(h_test_root, 1)

		true_label = test['root_l']

		if pred_label == true_label:
			acc += 1

		pred_label_li.append(pred_label)
		true_label_li.append(true_label)

		total += 1

	macro_f1 = f1_score(pred_label_li, true_label_li, average = 'macro')


	print(test_file, 'accuracy:', acc / total)
	print(test_file, 'f1:', macro_f1)
	print(test_file, 'total tested:', total)
