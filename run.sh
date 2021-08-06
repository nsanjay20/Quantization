# for MODEL in resnet18 resnet50 inceptionv3 mobilenetv2_w1 shufflenet_g1_w1 sqnxt23_w2 vgg19
for MODEL in resnet18 
 
#for MODEL in resnet20_cifar10
do
	echo Testing $MODEL ...
	python quantization_test.py 		\
		--dataset=imagenet 		\
		--model=$MODEL 			\
		--data-source 'train'   \
		--batch_size=64		\
		--test_batch_size=128 
done
