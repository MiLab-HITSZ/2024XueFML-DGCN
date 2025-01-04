
import math
import torch.nn
import torchnet.meter as tnt

import copy
import json
import os
import typing
from collections import OrderedDict
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.gcn.models import *
from federatedml.nn.backend.utils.APMeter import AveragePrecisionMeter
from federatedml.nn.backend.utils.aggregators.aggregator import *
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.param.gcn_param import GCNParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

cur_dir_name = os.getcwd()
my_writer = MyWriter(dir_name=cur_dir_name)

client_header = ['epoch', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map',
                 'loss']
server_header = ['agg_iter', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3',
                 'map', 'loss']
train_writer = my_writer.get("train.csv", header=client_header)
valid_writer = my_writer.get("valid.csv", header=client_header)
avgloss_writer = my_writer.get("avgloss.csv", header=server_header)


class _FedBaseContext(object):
    def __init__(self, max_num_aggregation, name):
        self._name = name

        self.max_num_aggregation = max_num_aggregation
        self._aggregation_iteration = 0

    def _suffix(self, group: str = "model"):
        return (
            self._name,
            group,
            f"{self._aggregation_iteration}",
        )

    def increase_aggregation_iteration(self):
        self._aggregation_iteration += 1

    @property
    def aggregation_iteration(self):
        return self._aggregation_iteration

    def finished(self):
        if self._aggregation_iteration >= self.max_num_aggregation:
            return True
        return False


class FedClientContext(_FedBaseContext):
    def __init__(self, max_num_aggregation, aggregate_every_n_epoch, name="feat"):
        super(FedClientContext, self).__init__(max_num_aggregation=max_num_aggregation, name=name)
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Client(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Client(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self._params: list = []

        self._should_stop = False
        self.loss_summary = []

    def init(self):
        self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    def send_model(self, tensors, bn_data, relation_matrix, weight):
        tensor_arrs = []
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
        bn_arrs = []
        for bn_item in bn_data:
            bn_arr = bn_item.data.cpu().numpy()
            bn_arrs.append(bn_arr)
        self.aggregator.send_model(
            (tensor_arrs, bn_arrs, relation_matrix, weight), suffix=self._suffix()
        )

    def recv_model(self):
        return self.aggregator.get_aggregated_model(suffix=self._suffix())

    def send_metrics(self, ap, mAP, loss, weight):
        self.aggregator.send_model((ap, mAP, loss, weight), suffix=self._suffix(group="metrics"))

    def do_aggregation(self, bn_data, relation_matrix, weight, device):

        self.send_model(self._params, bn_data, relation_matrix, weight)

        recv_elements: typing.List = self.recv_model()

        global_model, bn_data, relation_matrix = recv_elements
        agg_tensors = []
        for arr in global_model:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(self._params, agg_tensors):

            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

        bn_tensors = []
        for arr in bn_data:
            bn_tensors.append(torch.from_numpy(arr).to(device))
        return bn_tensors, relation_matrix

    def do_convergence_check(self, weight, ap, mAP, loss_value):
        self.send_metrics(ap, mAP, loss_value, weight)
        return False

    def configure_aggregation_params(self, optimizer):
        if optimizer is not None:
            self._params = [
                param
                for param_group in optimizer.param_groups
                for param in param_group["params"]
            ]
            return
        raise TypeError(f"params and optimizer can't be both none")

    def should_aggregate_on_epoch(self, epoch_index):
        return (epoch_index + 1) % self.aggregate_every_n_epoch == 0

    def should_stop(self):
        return self._should_stop

    def set_converged(self):
        self._should_stop = True


class FedServerContext(_FedBaseContext):
    def __init__(self, max_num_aggregation, eps=0.0, name="feat"):
        super(FedServerContext, self).__init__(
            max_num_aggregation=max_num_aggregation, name=name
        )
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Server(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Server(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self._eps = eps
        self._loss = math.inf

    def init(self, init_aggregation_iteration=0):
        self.random_padding_cipher.exchange_secret_keys()
        self._aggregation_iteration = init_aggregation_iteration

    def send_model(self, aggregated_arrs):

        self.aggregator.send_aggregated_model(aggregated_arrs, suffix=self._suffix())

    def recv_model(self):
        return self.aggregator.get_models(suffix=self._suffix())

    def send_convergence_status(self, mAP, status):
        self.aggregator.send_aggregated_model(
            (mAP, status), suffix=self._suffix(group="convergence")
        )

    def recv_metrics(self):
        return self.aggregator.get_models(suffix=self._suffix(group="metrics"))


def build_aggregator(param: GCNParam, init_iteration=0):
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    fed_aggregator = GCNFedAggregator(context)
    return fed_aggregator


def build_fitter(param: GCNParam, train_data, valid_data):

    category_dir = '/data/projects/fate/my_practice/dataset/coco2017/'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    context.init()
    inp_name = 'coco2017_glove_word2vec.pkl'
    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path, inp_name=inp_name)

    train_loader, valid_loader = dataset_loader.get_loaders(batch_size, dataset="COCO", drop_last=True,
                                                            num_workers=4)

    fitter = GCNFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader, 'normal'


class GCNFedAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context
        self.model = None
        self.bn_data = None
        self.relation_matrix = None

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()
            cur_iteration = self.context.aggregation_iteration
            tensors = [party_tuple[0] for party_tuple in recv_elements]
            bn_tensors = [party_tuple[1] for party_tuple in recv_elements]

            relation_matrices = [party_tuple[2] for party_tuple in recv_elements]
            degrees = [party_tuple[3] for party_tuple in recv_elements]
            self.bn_data = aggregate_bn_data(bn_tensors, degrees)

            self.relation_matrix = aggregate_relation_matrix(relation_matrices, degrees)
            self.model = aggregate_whole_model(tensors, degrees)

            self.context.send_model((self.model, self.bn_data, self.relation_matrix))


            np.save(f'{cur_dir_name}/global_model_{self.context.aggregation_iteration}', self.model)
            np.save(f'{cur_dir_name}/bn_data_{self.context.aggregation_iteration}', self.bn_data)
            np.save(f'{cur_dir_name}/relation_matrix_{self.context.aggregation_iteration}', self.relation_matrix)
            self.context.increase_aggregation_iteration()

    def export_model(self, param):
        pass

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        param.restore_from_pb(meta_obj.params)

    pass

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        pass

    @staticmethod
    def dataset_align():
        LOGGER.info("start label alignment")
        label_mapping = HomoLabelEncoderArbiter().label_alignment()
        LOGGER.info(f"label aligned, mapping: {label_mapping}")


class GCNFitter(object):
    def __init__(
            self,
            param,
            epochs,
            label_mapping=None,
            context: FedClientContext = None
    ):
        self.scheduler = ...
        self.param = copy.deepcopy(param)
        self._all_consumed_data_aggregated = True
        self.context = context
        self.label_mapping = label_mapping

        image_id2labels = json.load(open(self.param.adj_file, 'r'))
        num_labels = self.param.num_labels
        adjList = np.zeros((num_labels, num_labels))
        nums = np.zeros(num_labels)
        for image_info in image_id2labels:
            labels = image_info['labels']
            for label in labels:
                nums[label] += 1
            n = len(labels)
            for i in range(n):
                for j in range(i + 1, n):
                    x = labels[i]
                    y = labels[j]
                    adjList[x][y] += 1
                    adjList[y][x] += 1
        nums = nums[:, np.newaxis]
        for i in range(num_labels):
            if nums[i] != 0:
                adjList[i] = adjList[i] / nums[i]

        t = self.param.t
        adjList[adjList < t] = 0
        adjList[adjList >= t] = 1
        adjList = adjList * 0.25 / (adjList.sum(0, keepdims=True) + 1e-6)
        adjList = adjList + np.identity(num_labels, np.int)
        self.adjList = adjList

        self.model, self.scheduler, self.optimizer, self.gcn_optimizer = _init_gcn_learner(self.param,
                                                                                           self.param.device,
                                                                                           adjList)


        self.criterion = torch.nn.MultiLabelSoftMarginLoss().to(self.param.device)

        self.start_epoch, self.end_epoch = 0, epochs

        self._num_data_consumed = 0
        self._num_label_consumed = 0
        self._num_per_labels = [0] * self.param.num_labels

        self.ap_meter = AveragePrecisionMeter(difficult_examples=False)

        self.lr_scheduler = None

    def get_label_mapping(self):
        return self.label_mapping

    def fit(self, train_loader, valid_loader, agg_type):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.on_fit_epoch_start(epoch, len(train_loader.sampler))
            valid_metrics = self.train_validate(epoch, train_loader, valid_loader, self.scheduler)
            self.on_fit_epoch_end(epoch, valid_loader, valid_metrics)
            if self.context.should_stop():
                break

    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples

    def on_fit_epoch_end(self, epoch, valid_loader, valid_metrics):
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)

            self._all_consumed_data_aggregated = True

            self._num_data_consumed = 0
            self._num_label_consumed = 0
            self._num_per_labels = [0] * self.param.num_labels

            self.context.increase_aggregation_iteration()


    def train_one_epoch(self, epoch, train_loader, scheduler):

        self.ap_meter.reset()
        metrics = self.train(train_loader, self.model, self.criterion,
                             self.optimizer, epoch, self.param.device,
                             scheduler)
        return metrics

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        self.ap_meter.reset()
        metrics = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device, scheduler)
        return metrics

    def aggregate_model(self, epoch, weight=None):
        self.context.configure_aggregation_params(self.optimizer)

        bn_data = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_data.append(layer.running_mean)
                bn_data.append(layer.running_var)

        weight_list = list(self._num_per_labels)
        weight_list.append(self._num_data_consumed)

        agg_bn_data, adjList = self.context.do_aggregation(weight=weight_list, bn_data=bn_data,
                                                           device=self.param.device, relation_matrix=self.adjList)
        idx = 0
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.data.copy_(agg_bn_data[idx])
                idx += 1
                layer.running_var.data.copy_(agg_bn_data[idx])
                idx += 1
        self.model.updateA(adjList)

    def train_validate(self, epoch, train_loader, valid_loader, scheduler):
        self.train_one_epoch(epoch, train_loader, scheduler)
        valid_metrics = None
        if valid_loader:
            valid_metrics = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return valid_metrics

    def train(self, train_loader, model, criterion, optimizer, epoch, device, scheduler):
        self.ap_meter.reset()
        model.train()

        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        sigmoid_func = torch.nn.Sigmoid()

        for train_step, ((features, inp), target) in enumerate(train_loader):

            features = features.to(device)
            inp = inp.to(device)
            target = target.to(device)

            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            self._num_label_consumed += target.sum().item()


            output = model(features, inp)

            self.ap_meter.add(output.data, target)

            loss = criterion(output, target)
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        mAP, _ = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean

        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item(), loss]
        train_writer.writerow([epoch] + metrics)
        return metrics

    def validate(self, valid_loader, model, criterion, epoch, device, scheduler):
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        sigmoid_func = torch.nn.Sigmoid()
        model.eval()
        self.ap_meter.reset()

        with torch.no_grad():
            for validate_step, ((features, inp), target) in enumerate(valid_loader):
                features = features.to(device)
                inp = inp.to(device)
                target = target.to(device)

                output = model(features, inp)
                self.ap_meter.add(output.data, target)

                loss = criterion(output, target)

                losses[OBJECTIVE_LOSS_KEY].add(loss.item())
        mAP, _ = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item(), loss]
        valid_writer.writerow([epoch] + metrics)
        return metrics


def _init_gcn_learner(param, device='cpu', adjList=None):

    in_channel = 300

    model = resnet_c_gcn(param.pretrained, adjList=adjList,
                         device=param.device, num_classes=param.num_labels, in_channel=in_channel,
                         dataset=param.dataset, t=param.t)
    gcn_optimizer = None

    lr, lrp = param.lr, 0.1

    optimizer = torch.optim.SGD(model.get_config_optim(lr=lr, lrp=lrp),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    scheduler = None
    return model, scheduler, optimizer, gcn_optimizer
