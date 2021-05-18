CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset cifar10 \
--arch resnet_cifar \
--depth 18 \
--lr 0.1 \
--epochs 60 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 256 \
--save ./baseline/resnet18-cifar10 \
--momentum 0.9 \
--sparsity-regularization

