These are the training scripts for a [blog post that I wrote](https://codekansas.github.io/nlp-convs) about using convolutional neural networks for NLP. These scripts can be run from the command line, or using the recipes in the Makefile. The latter will handle installing the correct dependencies; it is assumed that your system has a GPU for training the models, so `tensorflow-gpu` is installed. You can change the `TENSORFLOW_PIP_VERSION` in the constants to `tensorflow` if you want to train without a GPU.

```bash
make binary   ; make productionize  # Creates the binary classification examples
make language ; make productionize  # Creates the language model examples
make clean                          # Removes any created directories
```
