python resprune.py \
--dataset cifar10 \
--arch resnet_cifar \
--depth 18 \
--test-batch-size 256 \
--depth 16 \
--percent 0.9 \
--model ./baseline/resnet18-cifar10/EB-90-07.pth.tar \
--save ./baseline/resnet18-cifar10/pruned_9007_0.9 \
--gpu_ids 1
