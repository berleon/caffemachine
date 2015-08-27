#! /usr/bin/env python3
# Copyright 2015 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import yaml
from . import Caffe, CaffeTemplate


def load_config_file(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    tmpl = CaffeTemplate(
        config['git_url'], config.get('git_tag'),
        allow_download_script=config.get("allow_download_script"))
    caffe = Caffe.get_caffe(config.get('caffe_git_tag'),
                            git_repo=config.get('caffe_git_url'))
    return caffe, tmpl, config['networks']


def train(args):
    caffe, tmpl, networks_cfg = load_config_file(args.config)
    for name, net_config, in networks_cfg.items():
        net = tmpl.find_or_render(name, net_config)
        caffe.train(net)


def evaluate(args):
    caffe, tmpl, networks_cfg = load_config_file(args.config)
    evaluates = []
    for name, net_config in networks_cfg.items():
        net = tmpl.find_or_render(name, net_config)
        if not net.weights():
            print("No weights found for the network `{}`.".format(name))
            print("You need to first train the networks before "
                  "evaluating them.")
            sys.exit(1)
        accuracy = 100*caffe.test(net, weights=net.highest_iteration_weights())
        time = caffe.time(net)
        evaluates.append((name, accuracy, time['avg_forward'],
                       time['avg_backward']))
    evaluates.sort(key=lambda s: s[0])
    print("{:^20}|{:^20}|{:^20}|{:^20}".format(
        "name", "accuracy [%]", "avg_forward [ms]", "avg_backward [ms]"))
    print(("-" * 20 + "+") * 3 + "-" * 20)
    for name, acc, avg_forward, avg_backward in evaluates:
        print("{:^20}|{:^20.3f}|{:^20.3f}|{:^20.3f}"
              .format(name, acc, avg_forward, avg_backward))


def extract(args):
    tmpl = CaffeTemplate(args.git_url)
    last_part = args.git_url.split("/")[-1]
    name = last_part + "_config"
    if args.tag:
        git_tag = args.tag
    else:
        git_tag = "master"
    config = {
        "name": last_part,
        "git_url": args.git_url,
        "git_tag": git_tag,
        "allow_download_script": False,
        "networks": {
            "default_network": {}
        }
    }
    var_obj = {}
    for var in tmpl.extract_variables():
        var_obj[var] = ""
    config['networks']['default_network'] = var_obj
    print(yaml.dump(config, default_flow_style=False, indent=2))


def arg_parser():
    parser = argparse.ArgumentParser(prog="caffemachine")
    subparsers = parser.add_subparsers()
    extract_parser = subparsers.add_parser(
        'extract', help='extract all variables in the template')
    extract_parser.add_argument("git_url", help='url of the git repository.')
    extract_parser.add_argument('--tag', help='git tag to checkout.')
    extract_parser.set_defaults(func=extract)

    train_parser = subparsers.add_parser('train', help='trains a network.')
    train_parser.add_argument('config', help='config file')
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser(
        'evaluate', help='evaluates the accuracy and the forward / backward '
                         'timings of the networks')
    evaluate_parser.add_argument('config', help='config file')
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
