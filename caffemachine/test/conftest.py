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
import pytest
from caffemachine import CaffeTemplate


@pytest.fixture
def test_tmpl() -> CaffeTemplate:
    git_repo = "file:///home/leon/uni/bachelor/coffe_machine_test"
    return CaffeTemplate(git_repo)

@pytest.fixture()
def net(test_tmpl) -> CaffeTemplate:
    config = {
        'conv1': {
            'num_output': 12,
            'kernel_size': 5
        },
        'conv2': {
            'num_output': 32,
            'kernel_size': 3
        }
    }
    return test_tmpl.render("test", config)

