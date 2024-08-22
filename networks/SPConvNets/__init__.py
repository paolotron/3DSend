import torch
import torch.nn as nn

def strip_checkpoint(checkpoint):
    checkpoint = checkpoint['model']
    new_ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('enco'):
            k = k[5:]
        elif k.startswith('module.enco'):
            k = k[12:]
        new_ckpt[k] = v
    return new_ckpt

class model_3dos(torch.nn.Module):

    def __init__(self, enco):
        super().__init__()
        self.enco = enco
        self.penultimate = nn.Sequential(
        nn.Linear(256, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256, bias=False)
        )
    def forward(self, x, penultimate=False):
        x = self.enco(x, penultimate=True)
        x = self.penultimate(x)
        if penultimate:
            return x
        return x



def get_model_openpatch(pretrained: str = ''):
    from .cls_so3net_pn import build_model as build_classifier
    from .reg_so3net import build_model as build_registrator
    from .inv_so3net_pn import build_model as build_inv
    strict = True
    if pretrained == 'modelnet':
        model = build_classifier(so3_pooling='attention')
        checkpoint = torch.load('./checkpoints/spconv_modelnet.pth')
    elif pretrained == '3dmatch':
        model = build_inv(1024)
        checkpoint = torch.load('./checkpoints/spconv_3dmatch.pth')
    elif pretrained == '3dmatch_small':
        model = build_classifier(so3_pooling='attention')
        checkpoint = torch.load('./checkpoints/spconv_small_3dmatch.pth')
        strict = False
    elif pretrained == 'objaverse':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls.pth')
    elif pretrained == 'objaverse_no_ag':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls_noag.pth')
    elif pretrained == 'objaverse_rot':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls_rot.pth')
    elif pretrained == 'objaverse_rot_salad':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls_rot_salad.pth')
    elif pretrained == 'objaverse_rot_salad_knn':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls_rotsalad_knn.pth')
    elif pretrained == 'objaverse_rot_salad_ball':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls_rotsalad_ball.pth')
    elif pretrained == 'objaverse_rot_salad_knn256':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls_rotsalad_knn256.pth')
    elif pretrained.startswith('supervised'):
        split = pretrained.split('_')[1].upper()
        model = build_classifier(so3_pooling='max', k=40)
        model = model_3dos(model)
        checkpoint = torch.load(f'./checkpoints/{split}_spconv_staug.pth')['model']
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        strict = False
    elif pretrained.startswith('finetuned'):
        split = pretrained.split('_')[1].upper()
        model = build_classifier(so3_pooling='max', k=40)
        model = model_3dos(model)
        checkpoint = torch.load(f'./checkpoints/{split}_spconvfine_staug.pth')['model']
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        strict = False
    elif pretrained == 'vicreg':
        model = build_classifier(so3_pooling='max', k=40)
        checkpoint = strip_checkpoint(torch.load(f'./checkpoints/vic_reg_spconv_objaverse.pth'))
        strict = False
    elif pretrained == 'objaverse_salad':
        model = build_classifier(so3_pooling='max', k=1156)
        raise NotImplementedError
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls.pth')
    elif pretrained == 'objaverse_no_overlap':
        model = build_classifier(so3_pooling='max', k=1156)
        checkpoint = torch.load('./checkpoints/spconv_objaverse_cls_nooverlap.pth')['model']
    else:
        raise NotImplementedError
    print(model)
    print(model.load_state_dict(checkpoint, strict=strict))

    return model
