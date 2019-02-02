# train
nohup python main.py --save_iters=10000 > log_train_HAN &

# test
nohup python main.py --mode='test' --test_iters=10000 > log_test_HAN &
