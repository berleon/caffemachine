# coffemachine

coffemachine is a utility program to generate, train and evaluate
[caffe](https://github.com/BVLC/caffe) models. Through the
[jinja2](http://jinja.pocoo.org/) template engine you can test make your
models prototxt files generic. It let's you test different number of neurons,
change the depth of your networks or the type of neurons.

## Install

```shell
$ git clone https://github.com/berleon/coffemachine.git
$ pip install caffemachine
```

## Create a caffe model template

`coffemachine` uses the [jinja2](http://jinja.pocoo.org/) template engine.
You can set the `num_outputs` of

### Repository structure

A caffe template is simply a git repository with the following structure:

```
/
├── deploy.prototxt.j2    Network description template ...
├── README.md
├── get_data.sh           Script to download the training and testset
├── solver.prototxt.j2    Solver description template for training the network
└── train.prototxt.j2     Network description template used for training and validation
```
Check this [simple MNIST](https://github.com/berleon/caffemachine_mnist.git) example out.

### Download data

To download the training / test data you can add a script to the template repository and
then

## Create a yml config file

Use the `caffemachine extract` command to generate a default config file:

```shell
$ caffemachine extract https://github.com/berleon/coffee_mnist.git
```
The output:
```yaml
allow_download_script:
git_tag: master
git_url: https://github.com/berleon/coffee_mnist.git
name: coffee_mnist
networks:
  default_network:
    conv1_num_output: ''
    conv2_num_output: ''
    ip1_num_output: ''
```
Nice it found all the variables used in the template!

Now edit it as you need. You might come up with something similar as the
[config_example.yml](https://github.com/berleon/caffemachine_mnist/blob/master/config_example.yml)
file from `coffee_mnist`:
```
name: lenet
git_url: https://github.com/berleon/coffee_mnist.git
git_tag: master
get_data_script: get_mnist.sh
networks:
  small:
    conv1_num_output: 13
    conv2_num_output: 35
    ip1_num_output: 250
  medium:
    conv1_num_output: 20
    conv2_num_output: 50
    ip1_num_output: 500
  big:
    conv1_num_output: 30
    conv2_num_output: 80
    ip1_num_output: 1000
```

Detailed description of the config file:
- **`git_url`**: `required` Url to the git repository of the template
- **`git_tag`**: `optional` Git tag to checkout. (default: master)
- **`allow_download_script`**: `optional` **Possible Harmfull** Do you allow the repository to run
  arbitrary code on your computer. It might download the training and testset or
  do something entire evil.
- **`caffe_git_url`**: `optional` URL to the git repository to caffe. (default: https://github.com/BVLC/caffe.git)
- **`caffe_git_tag`**: `optional` Caffe's git tag to use for compiling.
- **`networks`**: `required` A dictionary of `<network_name>`: {<variable>: <value>}.
  This will be used to initialize the templates.


If both `caffe_git_url` and `caffe_git_tag` are missing then `coffemachine`
will use the system wide `caffe` command. If the `caffe` command was not found than it will
download caffe and compile its master branch.


## Train

Trains all networks specified in `mnist_coffe.yml`.

```shell
$ caffemachine train mnist_coffe.yml
```

## Evaluate

Evaluates the accuracy and forward/backward timings of all networks in
`CONFIG_FILE`.

```shell
$ coffemachine evaluate CONFIG_FILE
```

```
        name        |    accuracy [%]    |  avg_forward [ms]  | avg_backward [ms]
--------------------+--------------------+--------------------+--------------------
        big         |       98.240       |       40.169       |       30.228
       medium       |       98.160       |       16.588       |       12.707
       small        |       97.640       |       10.519       |       7.559
```

## ToDo list

* Add `ssh` support: make it possible to run multiple networks parallel on
  different servers.
* Auto detect GPU support and use it if available
* Create a utility class to generate config files
* Select only a subset of networks to train / evaluate
* Save and analyse the trainings log and add a `report` subcommand
* Add flags to the config files to automatically download data. Without the
  running a potential evil script.
