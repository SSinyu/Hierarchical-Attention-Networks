import os
import argparse
from torch.backends import cudnn
from pytorch_loader import get_dataloader
from pytorch_solver import Solver


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.mode == 'train':
        train_loader = get_dataloader(data_path=args.data_path,
                                      max_sent=args.max_sent,
                                      max_doc=args.max_doc,
                                      mode=args.mode,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)

        eval_loader = get_dataloader(data_path=args.data_path,
                                     max_sent=args.max_sent,
                                     max_doc=args.max_doc,
                                     mode='eval',
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)

        solver = Solver(args, train_loader=train_loader, eval_loader=eval_loader)
        solver.train()

    elif args.mode == 'test':
        test_loader = get_dataloader(data_path=args.data_path,
                                     max_sent=args.max_sent,
                                     max_doc=args.max_doc,
                                     mode=args.mode,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)

        solver = Solver(args, test_loader=test_loader)
        solver.test()

    else:
        raise ValueError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--save_path', type=str, default='./save/')

    parser.add_argument('--max_sent', type=int, default=40)
    parser.add_argument('--max_doc', type=int, default=30)

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=6)

    parser.add_argument('--print_iters', type=int, default=10)
    parser.add_argument('--eval_iters', type=int, default=500)
    parser.add_argument('--decay_iters', type=int, default=5000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1000)
    
    parser.add_argument('--clip', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=.999)

    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    main(args)

