import os
import argparse
from tf_loader import text_dataloader
from tf_solver import Solver

def mkdir_(path_):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print('Create path : {}'.format(path_))


def main(args):
    mkdir_(args.save_path)

    if args.mode == 'train':
        train_loader = text_dataloader(data_path=args.data_path,
                                       max_sent=args.max_sent,
                                       max_doc=args.max_doc,
                                       mode=args.mode,
                                       batch_size=args.batch_size)
        eval_loader = text_dataloader(data_path=args.data_path,
                                      max_sent=args.max_sent,
                                      max_doc=args.max_doc,
                                      mode=args.mode,
                                      batch_size=args.batch_size)

        mkdir_(os.path.join(args.save_path, 'summary', 'train'))
        mkdir_(os.path.join(args.save_path, 'summary', 'eval'))

        solver = Solver(args, train_loader=train_loader, eval_loader=eval_loader)
        solver.train()

    elif args.mode == 'test':
        test_loader = text_dataloader(data_path=args.data_path,
                                      max_sent=args.max_sent,
                                      max_doc=args.max_doc,
                                      mode=args.mode,
                                      batch_size=args.batch_test)

        solver = Solver(args, test_loader=test_loader)
        solver.test(args.test_iters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='./')
    parser.add_argument('--save_path', type=str,
                        default='./save')

    parser.add_argument('--max_sent', type=int, default=40)
    parser.add_argument('--max_doc', type=int, default=30)

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_test', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--print_iters', type=int, default=10)
    parser.add_argument('--eval_iters', type=int, default=500)
    parser.add_argument('--decay_iters', type=int, default=5000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1000)

    parser.add_argument('--clip', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=.999)

    args = parser.parse_args()
    main(args)

