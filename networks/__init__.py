
import timm  # noqa
import torchvision.models as models  # noqa
try:
    import networks.SPConvNets as SPconv  # noqa
except ImportError:
    pass
try:
    import networks.OpenShape as OpenShape
    from huggingface_hub import hf_hub_download
except ImportError or ValueError:
    pass
try:
    import networks.uni3d as uni3d # noqa
except ImportError or ValueError:
    pass


_BACKBONES = {
    'epn': 'SPconv.get_model_openpatch(pretrained=None)',
    
    'OpenShape': 'OpenShape.load_model(model_name="OpenShape/openshape-spconv-all")',
    'OpenShape_ShapeNet': 'OpenShape.load_model(model_name="OpenShape/openshape-spconv-shapenet-only")',
    'OpenShape_Bert': 'OpenShape.load_bert(model_name="OpenShape/openshape-pointbert-vitg14-rgb")',
    'OpenShape_Bert_vitl14': 'OpenShape.load_model(model_name="OpenShape/openshape-pointbert-vitl14-rgb")',
    'OpenShape_Bert_vitb32': 'OpenShape.load_model(model_name="OpenShape/openshape-pointbert-vitb32-rgb")',
    'OpenShape_spconv': 'OpenShape.load_model()',
    
    'uni3d-b': "uni3d.get_uni3d_model(model_name='uni3d-b', model_zoo_path='%s')",
    'uni3d-b-no-lvis': "uni3d.get_uni3d_model(model_name='uni3d-b-no-lvis', model_zoo_path='%s')",
    'uni3d-l': "uni3d.get_uni3d_model(model_name='uni3d-l', model_zoo_path='%s')",
    'uni3d-l-no-lvis': "uni3d.get_uni3d_model(model_name='uni3d-l-no-lvis', model_zoo_path='%s')",
}



def load(name, model_zoo_path):
    command = _BACKBONES[name]
    if 'uni3d' in name:
        command = command % model_zoo_path
    return eval(command)

def get_1nn_layer(name):
    if 'spconv' in name:
        layer = 'outblock.pointnet'
    elif 'OpenShape' in name:
        layer = 'ppat'
    elif 'uni3d' in name:
        layer = 'point_encoder'
    return layer

def get_backbone(name: str, model_zoo_path: str):
    backbone = load(name, model_zoo_path)
    return backbone