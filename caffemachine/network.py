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

import glob
import json
import re

_iter_re = re.compile("(\d+)\.caffemodel")


class CaffeNet(object):
    def __init__(self, directory, template_args=None):
        self.directory = directory
        if template_args is None:
            with open(directory + "/template_args.json", "r") as f:
                self.template_args = json.load(f)
        else:
            self.template_args = template_args

    def solver_file(self):
        return self.directory + "/solver.prototxt"

    def train_file(self):
        return self.directory + "/train.prototxt"

    def test_file(self):
        return self.directory + "/deploy.prototxt"

    def template_args_file(self):
        return self.directory + "/template_args.json"

    @staticmethod
    def iteration_of_weights(caffemodel):
        match = re.search(_iter_re, caffemodel)
        return int(match.group(1))

    def weights(self):
        weights = glob.glob(self.directory + "/*.caffemodel")
        weights.sort(key=lambda caffemodel: self.iteration_of_weights(caffemodel))
        return weights

    def highest_iteration_weights(self):
        return self.weights()[-1]

