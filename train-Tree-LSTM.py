import torch

from treelstm import TreeLSTM, calculate_evaluation_orders, batch_tree_input, TreeDataset, convert_tree_to_tensors

from torch.utils.data import Dataset, IterableDataset, DataLoader

import os
import codecs

tree_path = '../../Dataset/Parsed-Trees/'

test_set = ['charliehebdo.txt'] #, 'germanwings-crash.txt', 'ottawashooting.txt', 'sydneysiege.txt']

IN_FEATURES = 40
OUT_FEATURES = 2
NUM_ITERATIONS = 100
BATCH_SIZE = 8

files = os.listdir(tree_path)

for test_file in test_set:
	# print('Now Testing ', test_file)
	print('Training Set:', set(files) - {test_file})

	test_trees = []
	train_trees = []

	for filename in files:
		input_file = codecs.open(tree_path + filename, 'r', 'utf-8')

		tree_li = []

		for row in input_file:
			s = row.strip().split('\t')

			tweet_id = s[0]
			curr_tree = eval(s[1])

			try:

				curr_tensor = convert_tree_to_tensors(curr_tree)

			except:
				continue

			tree_li.append(curr_tensor)

		input_file.close()

		if filename == test_file:
			test_trees = tree_li
		
		train_trees += tree_li
	
	model = TreeLSTM(IN_FEATURES, OUT_FEATURES).train()

	# loss_function = torch.nn.CrossEntropyLoss()
	loss_function = torch.nn.BCEWithLogitsLoss()

	optimizer = torch.optim.Adam(model.parameters())

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

		true_label = test['root_l'][0][1]

		if pred_label == true_label:
			acc += 1

		total += 1

	
	print(test_file, 'accuracy:', acc / total)