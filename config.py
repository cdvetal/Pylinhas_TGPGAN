EVOLUTION_TYPE = 'cmaes'

NUM_LINES = 7 # 13

SAVE_FOLDER = 'runs'
CHECKPOINT_FREQ = 1

#ALTERAR AQUI
POP_SIZE = 50 #32
N_GENS = 80  #gerações por passo
RESTART = 1 # 10 - numero de passos antes de reset do CMA
SIGMA = 0.1
EPOCHS = 10
SIZE_IMAGE = 28 # 28, 56, 84
GLYPH = "E"

LAMARCK = True
ADAM_STEPS = 0
LR = 1. # LR = 0.1 ou LR = 0.03

TARGET_FITNESS = 100000
RANDOM_SEED = None

FROM_CHECKPOINT = None  # None or "Experiment_name.pkl""

NETWORKS = ""  # mobilenet,vgg16
CLIP_MODEL = "ViT-B/32"

TARGET_CLASS = "birdhouse"

SAVE_ALL = False
VERBOSE = False

RENDERER = "pylinhas"#processing_python"

model_groups = {
    "london,": "xception,vgg16,vgg19,resnet50,resnet50v2,resnet101,resnet152,resnet101v2,resnet152v2,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,densenet121,densenet169,densenet201,nasnet,nasnetmobile,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7,",
    "standard3,": "vgg16,vgg19,mobilenet,",
    "standard6,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,",
    "standard9,": "standard6,inceptionresnetv2,nasnet,nasnetmobile,",
    "standard13,": "standard9,densenet121,densenet169,densenet201,mobilenetv2,",
    "standard18,": "standard13,resnet101,resnet152,resnet50v2,resnet101v2,resnet152v2,",
    "train1,": "vgg19,resnet50,inceptionv3,xception,",
    "standard,": "standard6,",
    "fantastic,": "inceptionv3,vgg16,xception,mobilenet,efficientnetb4,efficientnetb0",
    "all,": "standard18,",
}
