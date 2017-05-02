#!/bin/bash
declare -a learning_rates=(0.2 0.5)
declare -a hidden_sizes=(400)
declare -a keep_probs=(0.5 0.6 0.8)
declare -a lr_decays=(0.94 0.95 0.96)
for learning_rate in "${learning_rates[@]}"
do
	for hidden_size in "${hidden_sizes[@]}"
	do
		for keep_prob in "${keep_probs[@]}"
		do
			for lr_decay in "${lr_decays[@]}"
			do
				echo "Learning rate = $learning_rate"
				echo "hidden_size = $hidden_size"
				echo "keep_prob = $keep_prob"
				echo "lr_decay = $lr_decay"
				command="python main.py -f QSR -o test_batch_size 10 learning_rate $learning_rate hidden_size $hidden_size keep_prob $keep_prob lr_decay $lr_decay"
				#echo $command
				eval $command
			done
		done
	done
done
