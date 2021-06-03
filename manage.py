#!/usr/bin/env python3

import argparse
import subprocess as sp
from binance_prediction import __package_name__


def init(python_ver='3.8'):
    data = {'pn': __package_name__,
            'python_ver': python_ver}
    print('Project "{pn}" initialization has been started'.format(**data))
    result = sp.Popen('./bin/setup_env.sh {python_ver} {pn}'.format(**data), shell=True)
    result.communicate()
    returncode = result.returncode
    if returncode != 0:
        data['err_text'] = returncode
        raise RuntimeError('When installing the `{pn}` package, we got error'.format(**data))
    print('Project "{pn}" initialization has been finished'.format(**data))


def test(python_ver):
    data = {'pn': __package_name__,
            'python_ver': python_ver}
    print('Test of project "{pn}" has been started'.format(**data))
    sp.call(['./bin/test_project.sh {python_ver}'.format(**data)], shell=True)
    print('Test of project "{pn}" has been finished'.format(**data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', help='Name of function to call')
    parser.add_argument('-pv', '--python_ver', type=str, required=False, default='3.8', help='Python version (3, 3.8 ..)')
    args = parser.parse_args()
    globals()[args.function](args.python_ver)

