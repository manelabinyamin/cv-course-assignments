# cd icv83551/datasets
if [ ! -d "cifar-10-batches-py" ]; then
  curl http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz
  curl http://cs231n.stanford.edu/imagenet_val_25.npz
fi
# if [ ! -d "cifar-10-batches-py" ]; then
#   wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
#   tar -xzvf cifar-10-python.tar.gz
#   rm cifar-10-python.tar.gz
#   wget http://cs231n.stanford.edu/imagenet_val_25.npz
# fi
# cd ../..