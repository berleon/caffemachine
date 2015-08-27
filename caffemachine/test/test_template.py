import os


def test_template_git_clone(test_tmpl):
    train_file = os.path.expanduser(test_tmpl.template_dir +
                                    "/train.prototxt.j2")
    assert os.path.exists(train_file)


def test_template_init(test_tmpl, net):
    assert net.directory in test_tmpl.available_networks()


def test_config(net):
    assert os.path.exists(net.train_file())
    assert os.path.exists(net.test_file())
    assert os.path.exists(net.template_args_file())
