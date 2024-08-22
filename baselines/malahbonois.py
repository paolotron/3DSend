import argparse
import torch
import numpy as np
from tqdm import tqdm

from data.ood_metrics import calc_metrics
import networks
from tqdm import tqdm
from data.ood_datasets import set_train_loader_3d, set_test_loader_3d
import data.ood_metrics
import sys


def mahalanobis_evaluator(train_loader, test_loader, device, model, model_zoo_path):
    # implements neurips 2018: https://papers.nips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html

    # the mahalanobis distance is computed not for a single network layer, but for a
    # set of network layers (see model_feature_list function)

    # remember that this is not a per class mahalanobis, in the sense that we compute a single covariance matrix.
    # In practice we are assuming that the features covariance is the same for all classes, while of course the
    # features mean is different for each class.
    # This means that in practice we are not computing statistics over the whole dataset without
    # considering class labels. On the contrary we treat different classes separately, we compute for
    # them mean features, but for the covariance computation
    # we normalize each sample's features using the mean of its class. By doing this we
    # move all class clusters around the origin and build a single cluster
    # then we estimate the features covariance

    # original code print results for different noise magnitudes. We inherit the magnitude from ODIN
    epsilon = 0.001

    train_lbls = train_loader.dataset.dataset.labels
    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))


    if model.startswith('OpenShape'):
        train_loader.collate_fn = minkowski_collate
        t1, t2 = test_loader
        t1.collate_fn, t2.collate_fn = minkowski_collate, minkowski_collate
        test_loader = (t1, t2)
        extractor = model_feature_list_openshape
    elif  model.startswith('uni3d'):
        extractor = model_feature_list_uni3d
    
    model = networks.load(model, model_zoo_path=model_zoo_path).cuda()
    model = model.eval()

    with torch.no_grad():
        example_input, label = next(iter(train_loader))
        n_known_classes = len(known_labels)

        # first we need to know the number of channels in each network layer
        # we obtain it by a fake forward
        feature_list = np.array([el.shape[1] for el in extractor(model, example_input)])

        print('Estimate sample mean and covariance from train data')
        sample_mean, precision = sample_estimator(model, n_known_classes, feature_list, train_loader, device, model_type=model_type)
    # sample_mean, precision = [torch.rand(9, 256, device='cuda')], [torch.rand(256, 256, device='cuda')]
    print('get Mahalanobis scores')

    magnitude = epsilon

    # for each test sample we compute a normality score for each output layer of the network
    for i in range(len(feature_list)):
        m_score_in, test_labels_in = get_Mahalanobis_score(
            model,
            test_loader[0],
            n_known_classes,
            net_type="resnet",
            sample_mean=sample_mean,
            precision=precision,
            layer_index=i,
            magnitude=magnitude,
            device=device)
        m_score_out, test_labels_out = get_Mahalanobis_score(
            model,
            test_loader[1],
            n_known_classes,
            net_type="resnet",
            sample_mean=sample_mean,
            precision=precision,
            layer_index=i,
            magnitude=magnitude,
            device=device)
        m_score_in = np.asarray(m_score_in, dtype=np.float32)
        m_score_out = np.asarray(m_score_out, dtype=np.float32)
        m_score = np.concatenate([m_score_in, m_score_out])
        if i == 0:
            m_scores = m_score.reshape((m_score.shape[0], -1))
        else:
            m_scores = np.concatenate((m_scores, m_score.reshape((m_score.shape[0], -1))), axis=1)

    m_scores = np.asarray(m_scores, dtype=np.float32).T
    # ood_labels = prepare_ood_labels(known_labels.numpy(), test_labels.numpy())
    ood_labels = np.concatenate([np.ones(test_labels_in.shape[0]), np.zeros(test_labels_out.shape[0])])

    # print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_labels) - ood_labels.sum()}.")
    for l in range(len(m_scores)):
        metrics = calc_metrics(m_scores[l], ood_labels)
        print(f"Layer {l} auroc: {metrics['auroc']:.4f}, fpr95: {metrics['fpr_at_95_tpr']:.4f}")

    return calc_metrics(m_scores[-1], ood_labels)


# function to extract the multiple features
def model_feature_list_openshape(model, x):
    xyz, feats = x
    device = 'cuda'
    feats = model(xyz.float().to(device), feats.float().to(device), device=device, quantization_size=0.02)
    return [feats]

def model_feature_list_uni3d(model, x):
    x = x.float().cuda().transpose(2, 1)
    feats = model(x)
    return [feats]


def minkowski_collate(list_data, batch_coordinates=False, swap_axis=False):
    import MinkowskiEngine as ME
    xyz_list, feature_list, label_list = [], [], []
    for xyz, label in list_data:
        xyz = xyz.T
        if swap_axis:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        feats = np.concatenate([xyz, np.ones_like(xyz) * 0.4], axis=1)
        xyz_list.append(torch.tensor(xyz))
        label_list.append(torch.tensor(label))
        feature_list.append(torch.tensor(feats))
    if batch_coordinates:
        xyz = ME.utils.batched_coordinates(xyz_list, dtype=torch.float32)
        feats = torch.cat(feature_list)
        labels = torch.tensor(label_list)
    else:
        xyz = torch.tensor(np.array(xyz_list))
        feats = torch.tensor(np.array(feature_list))
        labels = torch.tensor(np.array(label_list))

    return (xyz, feats), labels


def sample_estimator(model, num_classes, feature_list, train_loader, device, model_type):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of per-class mean
            precision: list of precisions
    """
    import sklearn.covariance
    covariance_estimator = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    # prepare some data structures
    correct, total = 0, 0
    num_output_layers = len(feature_list)

    num_sample_per_class = np.zeros((num_classes))
    list_features = [[0] * num_classes for _ in range(len(feature_list))]

    if model_type == 'OpenShape':
        extractor = model_feature_list_openshape
    elif model_type == 'uni3d':
        extractor = model_feature_list_uni3d

    # for each model output layer, for each class, we extract for each sample
    # the per-channel mean feature value
    for batch in tqdm(train_loader):

        data, target = batch
        out_features = extractor(model, data)

        # compute the accuracy
        # correct += (output.argmax(1).cpu() == target).sum()

        # construct the sample matrix
        for i in range(1):
            label = target[i]
            # first sample for this class
            if num_sample_per_class[label] == 0:
                for layer_idx, layer_out in enumerate(out_features):
                    list_features[layer_idx][label] = layer_out[i].view(1, -1)
            else:
                for layer_idx, layer_out in enumerate(out_features):
                    list_features[layer_idx][label] = torch.cat(
                        (list_features[layer_idx][label], layer_out[i].view(1, -1)))
            num_sample_per_class[label] += 1

    sample_class_mean = []
    # for each output and for each class we compute the mean for each feat
    for layer_idx, layer_out in enumerate(feature_list):
        per_class_mean = torch.zeros((num_classes, layer_out)).to(device)
        for cls in range(num_classes):
            per_class_mean[cls] = torch.mean(list_features[layer_idx][cls], dim=0)
        sample_class_mean.append(per_class_mean)

    # we have computed features mean separately for each output layer and each class.
    # now for each output layer we want to estimate the covariance matrix (and compute the inverse)
    # thus we move all samples of a lyer around the origin, by normalizing through their class mean
    # then we compute the covariance and its inverse
    precision = []
    for k in range(num_output_layers):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        covariance_estimator.fit(X.cpu().numpy())
        temp_precision = covariance_estimator.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    # print('Training Accuracy:({:.2f}%)'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, net_type, sample_mean, precision, layer_index, magnitude,
                          device):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index

    Similarly to what is done for ODIN, a first forward is computed,
    then the predicted class for each sample is obtained
    (by looking at the mahalnobis distance from all classes),

    At this point a loss is computed using the predicted class as GT,
    the gradient on the input is used to apply a random noise on the input itself.
    This strategy should further separate known samples from unknown ones.

    The normality score is later computed using the corrupted input
    '''
    mahalanobis = []
    gt_labels = []
    device = 'cuda'
    for batch in tqdm(test_loader):
        data, target = batch
        gt_labels.append(target)

        # data.requires_grad = True
        xyz, feats = data
        feats.requires_grad = True

        out_features = model(xyz.float().to(device), feats.float().to(device), device=device, quantization_size=0.02)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)

        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(feats.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        tempFeats = torch.add(feats, gradient, alpha=-magnitude)
        tempFeats = tempFeats.detach()

        # perform forward again
        noise_out_features = model(xyz.float().to(device), tempFeats.float().to(device), device=device,
                                   quantization_size=0.02)

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        mahalanobis.extend(noise_gaussian_score.cpu().numpy())

    return mahalanobis, torch.cat(gt_labels)


if __name__ == '__main__':
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--in_dataset', type=str)
    parser.add_argument('--backbone', type=str, default='OpenShape_Bert')
    parser.add_argument('-bs', '--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_points', default=2048, type=int)
    parser.add_argument('--model_zoo_path', type=str)
    args = parser.parse_args()
    

    train_loader = set_train_loader_3d(data_root=args.data_path, in_dataset=args.in_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, num_points=args.num_points, split_classes=False,
                                        sparse=False, padding=0, seed_dataset=args.seed)
    test_loader = set_test_loader_3d(data_root=args.data_path, in_dataset=args.in_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, num_points=args.num_points, split_classes=False,
                                        sparse=False, padding=0, seed_dataset=args.seed)
    results = mahalanobis_evaluator(train_loader=train_loader, test_loader=test_loader, device='cuda',
                                    model=args.backbone, model_zoo_path=args.model_zoo_path)
    print(results)
    

