from tensorflow.keras.applications import Xception,VGG16,VGG19,ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2
from tensorflow.keras.applications import ResNet152V2,InceptionV3,MobileNet,MobileNetV2,DenseNet121,DenseNet169
from tensorflow.keras.applications import DenseNet201,NASNetMobile,EfficientNetB0,EfficientNetB1,EfficientNetB2


def return_model(model_name):
    if model_name == 'Xception':
        print("Loading Xception..")
        core_model = Xception(include_top=False,weights="imagenet",input_shape=(299, 299, 3))

    elif model_name == 'VGG16':
        print("Loading VGG16..")
        core_model = VGG16(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'VGG19':
        print("Loading VGG19..")
        core_model = VGG19(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'ResNet50':
        print("Loading ResNet50..")
        core_model = ResNet50(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'ResNet101':
        print("Loading ResNet101..")
        core_model = ResNet101(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'ResNet152':
        print("Loading ResNet152..")
        core_model = ResNet152(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'ResNet50V2':
        print("Loading ResNet50V2..")
        core_model = ResNet50V2(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'ResNet101V2':
        print("Loading ResNet101V2..")
        core_model = ResNet101V2(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'ResNet152V2':
        print("Loading ResNet152V2..")
        core_model = ResNet152V2(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'InceptionV3':
        print("Loading InceptionV3..")
        core_model = InceptionV3(include_top=False,weights="imagenet",input_shape=(299, 299, 3))

    elif model_name == 'MobileNet':
        print("Loading MobileNet..")
        core_model = MobileNet(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'MobileNetV2':
        print("Loading MobileNetV2..")
        core_model = MobileNetV2(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'DenseNet121':
        print("Loading DenseNet121..")
        core_model = DenseNet121(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'DenseNet169':
        print("Loading DenseNet169..")
        core_model = DenseNet169(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'DenseNet201':
        print("Loading DenseNet201..")
        core_model = DenseNet201(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'NASNetMobile':
        print("Loading NASNetMobile..")
        core_model = NASNetMobile(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'EfficientNetB0':
        print("Loading EfficientNetB0..")
        core_model = EfficientNetB0(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    elif model_name == 'EfficientNetB1':
        print("Loading EfficientNetB1..")
        core_model = EfficientNetB1(include_top=False,weights="imagenet",input_shape=(224, 224, 3))

    else:
        print("Loading EfficientNetB2..")
        core_model = EfficientNetB2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    return core_model

