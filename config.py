import shutil
import os
import argparse
import sys
sys.path.insert(0, '../')
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General settings
parser.add_argument('--model', type=str, default='gan', help="Model")
parser.add_argument('--design', type=str, default='improved', help="Design")
parser.add_argument('--dense', type=int, default=0, help="Densify")
parser.add_argument('--run', type=int, default=999, help="Run index.")
parser.add_argument('--mse', type=int, default=0, help="Learning rate.")
parser.add_argument('--trg-w', type=float, default=15, help="Learning rate.")
parser.add_argument('--src-w', type=float, default=15, help="Learning rate.")
parser.add_argument('--glr', type=float, default=3e-4, help="Learning rate.")
parser.add_argument('--dlr', type=float, default=3e-4, help="Learning rate.")

if 'ipykernel' in sys.argv[0]:
    parser.set_defaults(run=999, seed=999)
    args = parser.parse_args([])

else:
    args = parser.parse_args()

print args

def delete_existing(path):
    if args.run < 999:
        assert not os.path.exists(path)

    else:
        if os.path.exists(path):
            shutil.rmtree(path)

def get_log_dir():
    setup = [('model={:s}',  args.model),
             ('design={:s}',  args.design),
             ('dense={:d}',  args.dense),
             ('mse={:d}',  args.mse),
             ('trg_w={:02.0f}', args.trg_w),
             ('src_w={:02.0f}', args.src_w),
             ('glr={:.0e}',  args.glr),
             ('dlr={:.0e}',  args.dlr),
             ('run={:d}',     args.run)]

    log_dir = ''
    for template, val in setup:
        log_dir += template.format(val) + '_'

    log_dir = log_dir.rstrip('_')
    log_dir = os.path.join('log', log_dir)
    return log_dir
