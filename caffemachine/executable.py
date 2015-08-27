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
import subprocess
from subprocess import PIPE
import re
import sys
import shutil
from caffemachine import CACHE_DIR, CAFFE_CACHE_DIR
from .network import CaffeNet


def _die_if_fails(command, **kwargs):

    ret_code = subprocess.call(command, **kwargs)
    if ret_code != 0:
        print("""subprocess.call failed:
    command: `{}`
    arguments: {}""".format(command, kwargs))


class Caffe(object):
    def __init__(self, executable="caffe", caffe_ld_path=None, gpus=None):
        self.caffe_ld_path = caffe_ld_path
        self.executable = executable
        self.gpus = gpus

    def set_gpus(self, gpus):
        self.gpus = gpus

    def _get_gpus_as_str(self):
        if type(self.gpus) == tuple:
            return ",".join([str(g) for g in self.gpus])
        if self.gpus is None:
            return
        else:
            return str(self.gpus)

    _re_forward_layer = re.compile("]\s+(\w+)\tforward: ([\d\.]+)")
    _re_backward_layer = re.compile("]\s+(\w+)\tbackward: ([\d\.]+)")
    _re_avg_forward_layer = re.compile("] Average Forward pass: ([\d\.]+)")
    _re_avg_backward_layer = re.compile("] Average Backward pass: ([\d\.]+)")
    _re_train_loss = re.compile("Iteration (\d+), loss = ([\d\.]+)")
    _re_learning_rate = re.compile("Iteration (\d+), lr = ([\d\.]+)")
    _re_train_accuracy = re.compile(
        "Iteration (\d+)(.+)\n.*accuracy = ([\d\.]+)", re.MULTILINE)
    _re_test_accuracy = re.compile(
        "Iteration (\d+), Testing net(.+)\n.*accuracy = ([\d\.]+)",
        re.MULTILINE)
    _re_test_loss = re.compile(
        "Iteration (\d+), Testing net (.+)\n.*\n.*loss = ([\d\.]+)",
        re.MULTILINE)

    def _get_timings(self, log):
        layers = {}
        for match in re.finditer(self._re_forward_layer, log):
            layer_name = match.group(1)
            forward_time = match.group(2)
            layers[layer_name] = {'forward': float(forward_time)}
        for match in re.finditer(self._re_backward_layer, log):
            layer_name = match.group(1)
            backward_time = match.group(2)
            layers[layer_name]['backward'] = float(backward_time)
        match = re.search(self._re_avg_forward_layer, log)
        avg_forward = float(match.group(1))
        match = re.search(self._re_avg_backward_layer, log)
        avg_backward = float(match.group(1))
        return {
            'layers': layers,
            'avg_forward': avg_forward,
            'avg_backward': avg_backward
        }

    def time(self, net: CaffeNet, iterations=10, use_train_model=False):
        if use_train_model:
            model_file = net.train_file()
        else:
            model_file = net.test_file()
        args = [self.executable,  "time", "-iterations", str(iterations),
                "-model", model_file]
        if self.gpus is not None:
            args.extend(["-gpu", self._get_gpus_as_str()])
        p = self._run_caffe(args)
        stdout, stderr = p.communicate()
        log = stderr.decode("utf-8")
        if p.wait() != 0:
            print(log)
            print("Command failed: {}".format(" ".join(args)))
            sys.exit(1)
        report = self._get_timings(log)
        report['log'] = log
        return report

    def train(self, net: CaffeNet, snapshot=None):
        args = [self.executable,  "train", "-solver", net.solver_file()]
        if snapshot is not None:
            args.extend(["-snapshot", snapshot])
        if self.gpus is not None:
            args.extend(["-gpu", self._get_gpus_as_str()])
        p = self._run_caffe(args, cwd=net.directory)
        stderr_lines = []
        for byte_line in iter(p.stderr.readline, b''):
            line = byte_line.decode('utf-8')
            stderr_lines.append(line)
            print(line.rstrip())
        stderr = "".join(stderr_lines)
        if p.wait() != 0:
            print(stderr, file=sys.stderr)

    def test(self, net, weights, gpu=False, iterations=None):
        args = [self.executable, "test", "-model", net.train_file(),
                "-weights", weights]
        if iterations is not None:
            args.extend(["-iterations", iterations])
        if self.gpus is not None:
            args.extend(["-gpu", self._get_gpus_as_str()])

        p = self._run_caffe(args, cwd=net.directory)
        stdout, stderr = p.communicate()
        assert p.wait() == 0
        log = stderr.decode("utf-8")
        accuracy_re = re.compile("] accuracy = (.*)")
        matches = accuracy_re.search(log)
        accuracy = float(matches.group(1))
        return accuracy

    @classmethod
    def get_caffe(cls, git_tag=None, git_repo=None, **compile_opts):
        if git_tag is None and git_repo is None:
            caffe_system_exe = shutil.which("caffe")
            if caffe_system_exe:
                return cls(caffe_system_exe)
        if git_tag is None:
            git_tag = "master"
        if git_repo is None:
            git_repo = "https://github.com/BVLC/caffe.git"
        cache_repo = os.path.join(CAFFE_CACHE_DIR, git_tag)
        if not os.path.exists(cache_repo):
            _die_if_fails("git clone {} {}".format(git_repo, cache_repo),
                          shell=True)
        return cls.compile_caffe_if_not_avialable(cache_repo, **compile_opts)

    @classmethod
    def compile_caffe_if_not_avialable(cls, repo_path, blas="open", cpu_only=True):
        caffe_executable = os.path.join(repo_path, "build/install/bin/caffe")
        caffe_ld_path = os.path.join(repo_path, "build/install/lib/")
        if os.path.exists(caffe_executable):
            return Caffe(caffe_executable, caffe_ld_path=caffe_ld_path)
        build_dir = repo_path + "/build"
        os.makedirs(build_dir, exist_ok=True)
        cpu_only_str = ""
        if cpu_only:
            cpu_only_str = "-DCPU_ONLY=ON"

        cmake_cmd = "cmake -DBLAS={} {cpu} -DCAMKE_BUILD_TYPE=Release .." \
            .format(blas, cpu=cpu_only_str)
        _die_if_fails(cmake_cmd, cwd=build_dir, shell=True)
        _die_if_fails("make && make install", cwd=build_dir, shell=True)
        assert os.path.exists(caffe_executable)
        return cls(caffe_executable, caffe_ld_path=caffe_ld_path)

    def _run_caffe(self, args, **subprocess_popen_opts):
        env = dict(os.environ)
        if self.caffe_ld_path:
            env['LD_LIBRARY_PATH'] = self.caffe_ld_path
        return subprocess.Popen(args, stdout=PIPE, stderr=PIPE, env=env,
                                **subprocess_popen_opts)

    def get_most_accurate(self, nets):
        accuracies = [self.test(net, net.highest_iteration_weights()) for net in nets]
        max_index = accuracies.index(max(accuracies))
        return nets[max_index]

    def get_fastest(self, nets):
        timings = [self.time(net, net.highest_iteration_weights()) for net in nets]
        max_index = timings.index(max(timings, key='avg_forward'))
        return nets[max_index]
