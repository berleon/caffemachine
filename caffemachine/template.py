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
import hashlib
import json
import copy
import os
import sys
import subprocess
from subprocess import PIPE
import shutil

from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
import jinja2
import jinja2.meta

from . import TEMPLATE_CACHE_DIR, NETWORKS_DIR
from .network import CaffeNet


class CaffeTemplate(object):
    def __init__(self, git_url, git_tag=None, allow_download_script=None):
        if git_tag is None:
            git_tag = "master"
        if allow_download_script is None:
            self.allow_download_script = False
        else:
            self.allow_download_script = allow_download_script

        self.template_dir = self.cache_dir(git_url, git_tag)
        self.networks_dir = self.cache_dir(git_url, git_tag, for_networks=True)
        self._clone(git_url, git_tag)
        data_dir = self.data_dir()
        get_data_script = self.get_data_script()
        if os.path.exists(get_data_script) and not os.path.isdir(data_dir):
            self._run_get_data_script()

    def get_data_script(self):
        return os.path.join(self.template_dir, "get_data.sh")

    def data_dir(self):
        return os.path.join(self.template_dir, "data")

    def available_networks(self):
        networks = next(os.walk(self.networks_dir))[1]
        return [os.path.join(self.networks_dir, net) for net in networks]

    def extract_variables(self):
        env = Environment(loader=FileSystemLoader(self.template_dir))
        variables = set()
        templates_to_visit = list(self.template_files())
        while templates_to_visit:
            template = templates_to_visit.pop(0)
            template_source = env.loader.get_source(env, template)[0]
            ast = env.parse(template_source)
            templates_to_visit.extend(
                list(jinja2.meta.find_referenced_templates(ast)))
            variables.update(jinja2.meta.find_undeclared_variables(ast))
        return variables

    def template_files(self):
        for tmpl in glob.glob(self.template_dir + "/*.j2"):
            yield os.path.basename(tmpl)

    def _render_files(self, template_args, output_dir):
        env = SandboxedEnvironment(
            loader=FileSystemLoader(self.template_dir),
            extensions=[],
        )
        try:
            for template in self.template_files():
                output_str = env.get_template(template).render(template_args)
                output_file = output_dir + "/" + template.rstrip(".j2")
                with open(output_file, "w+") as f:
                    f.write(output_str)
        except TemplateSyntaxError as e:
            print("[{}:{}] {} ".format(e.filename, e.lineno, e.message))
            sys.exit(1)

    def _copy_files(self, output_dir):
        data_dir = os.path.join(self.template_dir, "data")
        output_data_dir = os.path.join(output_dir, "data")
        for f in os.listdir(self.template_dir):
            f_in_template = os.path.join(self.template_dir, f)
            f_in_network = os.path.join(output_dir, f)
            if os.path.isfile(f_in_template) and \
                    not f_in_template.endswith(".j2") and \
                    not os.path.exists(f_in_network):
                shutil.copyfile(f_in_template, f_in_network)
        if os.path.isdir(data_dir) and not os.path.exists(output_data_dir):
            os.symlink(data_dir, output_data_dir)

    def _network_dir(self, name, template_args):
        args_json_bytes = json.dumps(template_args,
                                     sort_keys=True).encode('utf-8')
        args_sha1 = hashlib.sha1(args_json_bytes).hexdigest()
        return os.path.join(self.networks_dir, name + '_' + args_sha1[:32])

    def find_or_render(self, name, template_args):
        output_dir = self._network_dir(name, template_args)
        if not os.path.exists(output_dir):
            self.render(name, template_args)
        else:
            return CaffeNet(output_dir, template_args=template_args)

    def render(self, name, template_args) -> CaffeNet:
        output_dir = self._network_dir(name, template_args)
        os.makedirs(output_dir, exist_ok=True)
        self._copy_files(output_dir)
        self._render_files(template_args, output_dir)
        with open(output_dir + "/template_args.json", "w") as c:
            json.dump(template_args, c, indent=4)
        return CaffeNet(output_dir, template_args=template_args)

    @staticmethod
    def cache_dir(git_url, git_tag, for_networks=False):
        parts = git_url.split("/")
        prefix = "-".join(parts[-2:])
        combined_git_url = git_url + "#" + git_tag
        suffix = hashlib.sha1(combined_git_url.encode('utf-8')).hexdigest()
        if for_networks:
            return os.path.join(NETWORKS_DIR, prefix + "-" + suffix[:32])
        else:
            return os.path.join(TEMPLATE_CACHE_DIR, prefix + "-" + suffix[:32])

    def _clone(self, git_url, git_tag=None):
        def run(cmd, **kwargs):
            return subprocess.call(cmd, shell=True,
                                   stderr=PIPE, stdout=PIPE, **kwargs)
        if not os.path.exists(self.template_dir + "/.git"):
            git_clone_cmd ="git clone {} {}".format(git_url, self.template_dir)
            assert run(git_clone_cmd) == 0, \
                "Failed to clone git repo"

        if git_tag is not None:
            git_checkout_cmd ="git checkout {}".format(git_tag)
            assert run(git_checkout_cmd, cwd=self.template_dir) == 0, \
                "Failed to clone git repo"

    def _print_security_warning(self):
        print("*" * 80)
        with open(self.get_data_script(), "r") as f:
            print(f.read())
        print("*" * 80)
        print("This template needs to download some data from the internet.")
        answer = ""
        while answer != "Y":
            print("This is potential harmful! "
                  "Do you trust the code above? [Y/n] ", end="", flush=True)
            answer = sys.stdin.readline().rstrip('\n')
            if answer == "n":
                print("You decided to not run the download script. Aborting!")
                sys.exit(1)

    def _run_get_data_script(self):
        if not self.allow_download_script:
            self._print_security_warning()
        subprocess.check_call(self.get_data_script(), cwd=self.template_dir)


