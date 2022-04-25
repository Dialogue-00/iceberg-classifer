source /home/lst/anaconda3/bin/activate
# conda activate pt
conda activate py3.8
# python main.py -m resnet18 -p False -e 30 -l 2e-4
# python main.py -m resnet34 -p False -e 50 -l 1e-4
# python main.py -m resnet34 -p True -e 30 -l 1e-4

# python main.py -b 32 -m resnet50 -p True -e 30 -l 1e-4
# python main.py -b 32 -m resnet50 -p False -e 30 -l 1e-4

# python main.py -b 8 -m vgg16 -p True -e 30 -l 1e-4
# python main.py -b 8 -m vgg19 -p False -e 30 -l 5e-3

# python main.py -b 32 -m alexnet -p True -e 30 -l 1e-4
# python main.py -b 32 -m alexnet -p False -e 30 -l 1e-4


# python main.py -b 16 -m vit -p True -e 30 -l 1e-4


# python main.py -b 32 -m resnet18 -p False -e 10 -l 1e-3
# python main.py -b 32 -m resnet18 -p True -e 10 -l 1e-3

# python main.py -b 32 -m vit -e 30 -l 1e-4

# python main.py -m resnet34 -p False -e 10 -l 1e-4
# python main.py -m resnet34 -p True -e 10 -l 1e-4


python main.py

