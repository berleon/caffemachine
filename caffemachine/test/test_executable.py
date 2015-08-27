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

import os

import pytest
import subprocess
from caffemachine import Caffe

@pytest.fixture
def caffe():
    return Caffe.get_caffe("rc2")


@pytest.mark.slow
def test_caffe_compile(caffe):
    assert os.path.exists(caffe.executable)
    assert subprocess.call([caffe.executable, "--version"]) == 0


def test_caffe_time(net):
    deploy_timings = Caffe().time(net, iterations=1)
    train_timings = Caffe().time(net, iterations=1, use_train_model=True)
    assert type(deploy_timings['avg_forward']) == float
    assert type(train_timings['avg_backward']) == float


def test_caffe_train(net):
    Caffe().train(net)
    assert len(net.weights()) > 0
    print(net.weights())
    assert net.iteration_of_weights(net.highest_iteration_weights()) == 1000


def test_caffe_test(net):
    caffe = Caffe()
    caffe.train(net)
    for w in net.weights():
        assert caffe.test(net, w) > 0.96
