import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable, Dict, Tuple
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from torchvision import transforms as T
import sys
import os
from contextlib import contextmanager
import numpy as np
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        # features = []
        # for model in self.extractors:
        #     features.append(model(x).squeeze())
        # features = torch.cat(features, dim=0)
        features = {}  # 不拼接,改为字典存储
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features

class EnsembleFeatureExtractor_ot(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5):
        super(EnsembleFeatureExtractor_ot, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_local = {}
        features_raw = {}  # 原始 patch embeddings(不聚类)
        for i, model in enumerate(self.extractors):
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            # 聚类中心(用于 OT 风格的损失)
            cluster_center = self.get_cluster_center(x_embedding[0], x.device).unsqueeze(0)
            features_local[i] = cluster_center
            # 原始 patch embeddings(用于 saliency loss)
            features_raw[i] = x_embedding.squeeze(0)  # [N_patches, D]

        return features, features_local, features_raw

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def forward_raw(self, x: Tensor) -> Tuple[Dict, Dict, Dict]:
        """返回包含原始 patch embeddings 的版本"""
        return self.forward(x)

    def intermediate_forward(self, x: Tensor) -> Tuple[Dict[int, List[Tensor]], Dict[int, List[Tensor]]]:
        """Return intermediate layer features for all models.

        Returns:
            all_global: dict {model_idx: list of [B, D] CLS tokens per layer}
            all_local:  dict {model_idx: list of [B, N, D] patch tokens per layer}
        """
        all_global = {}
        all_local = {}
        for i, model in enumerate(self.extractors):
            _, all_global_i, all_local_i = model.intermediate_features(x.to(x.device))
            all_global[i] = all_global_i
            all_local[i] = all_local_i
        return all_global, all_local

class EnsembleFeatureExtractor_ot3(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor_ot3, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        # features = []
        # for model in self.extractors:
        #     features.append(model(x).squeeze())
        # features = torch.cat(features, dim=0)
        features = {}  # 不拼接,改为字典存储
        features_local = {}
        for i, model in enumerate(self.extractors):
            # features[i] = model(x).squeeze()
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            cluster_center = self.get_cluster_center(x_embedding[0],x.device).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

    def get_cluster_center(self, embedding_img,device):
        # self.setup_seed(20)
        # for i in range(10):
        # np.random.seed(20)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=3,
                distance='euclidean',
                device=device
            )
            cluster_center = cluster_center.to(device)
        # print(cluster_ids_x)
        return cluster_center


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))

        loss = loss / len(self.extractors)

        return loss


class EnsembleFeatureLoss_OT(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []



    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)



    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        loss_local=0
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            # print("gt_local",gt_local.shape)
            # print("feature_local", feature_local.shape)
            loss_local += self.OT(gt_local, feature_local) * 2

            loss += torch.mean(torch.sum(feature * gt, dim=1))

        loss = loss / len(self.extractors)
        loss_local=loss_local / len(self.extractors)

        # print("loss",loss)
        # print("loss_local", loss_local)
        loss=loss+loss_local*0.1
        return loss

    def get_cluster_center(self, embedding_img):
        # self.setup_seed(20)
        # for i in range(10):
        # np.random.seed(20)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=5,
                distance='euclidean',
                device=torch.device('cuda')            )
            cluster_center = cluster_center.to('cuda')
        # print(cluster_ids_x)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T


class EnsembleFeatureLoss_OT_Auto(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT_Auto, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []



    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)



    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            # print("gt_local",gt_local.shape)
            # print("feature_local", feature_local.shape)
            local_loss = self.OT(gt_local, feature_local) * 2
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.1 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        # 初始化 previous_loss_list(首次)
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            # 归一化 softmax 计算动态权重
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  # 可选:缩放为 K


        # 初始化 previous_loss_list(首次)
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img):
        # self.setup_seed(20)
        # for i in range(10):
        # np.random.seed(20)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=5,
                distance='euclidean',
                device=torch.device('cuda')            )
            cluster_center = cluster_center.to('cuda')
        # print(cluster_ids_x)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_foa_attack(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5):
        super(EnsembleFeatureLoss_OT_foa_attack, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feature_local = feature_local_dict[index].squeeze(0)
            # print("gt_local",gt_local.shape)
            # print("feature_local", feature_local.shape)
            local_loss = self.OT(gt_local, feature_local)

            # feat_loss = torch.mean(torch.sum(feature * gt, dim=1))
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        # 初始化 previous_loss_list(首次)
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            # 归一化 softmax 计算动态权重

        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  # 可选:缩放为 K


        # 初始化 previous_loss_list(首次)
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        # self.setup_seed(20)
        # for i in range(10):
        # np.random.seed(20)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number, # TODO
                distance='euclidean',
                device=device            )
            cluster_center = cluster_center.to(device)
        # print(cluster_ids_x)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_ablation_wo_global(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_global, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local)
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        # 初始化 previous_loss_list(首次)
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            # 归一化 softmax 计算动态权重

        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  # 可选:缩放为 K


        # 初始化 previous_loss_list(首次)
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number, # TODO
                distance='euclidean',
                device=device            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_ablation_wo_local(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):

            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]
        # 初始化 previous_loss_list(首次)
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            # 归一化 softmax 计算动态权重

        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  # 可选:缩放为 K


        # 初始化 previous_loss_list(首次)
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number, # TODO
                distance='euclidean',
                device=device            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_ablation_wo_dynamic(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_dynamic, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):

            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]

        # 加权总损失
        total_loss = sum(total_losses)/len(total_losses)
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number, # TODO
                distance='euclidean',
                device=device            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_Gram(nn.Module):
    """
    FOA-Attack 原有 OT loss(全局 + 局部聚类 OT)的基础上,
    新增 Gram Matrix 风格损失:对齐 patch 特征的二阶统计量,
    实现整体纹理/风格/大局迁移。

    接口与 EnsembleFeatureLoss_OT_foa_attack 完全兼容,
    额外提供 gram_loss() 方法供外部计算 Gram MSE 项。
    """

    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number: int = 10):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.ground_truth_gram = []
        self.previous_loss_list = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        self.ground_truth_gram.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding, x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)
            _, gram = model.spatial_gram_features(x.to(x.device))
            self.ground_truth_gram.append(gram)

    def __call__(self, feature_dict: Dict[int, Tensor], feature_local_dict: Dict[int, Tensor],
                 y: Any = None) -> Tensor:
        """返回总 OT loss(全局 OT + 0.2*局部 OT),供 FOA Attack 优化用。"""
        loss_list, loss_local_list = [], []

        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feature_local = feature_local_dict[index].squeeze(0)

            local_loss = self.OT(gt_local, feature_local)
            feat_loss = self.OT(gt, feature)

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]

        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)

        T_softmax = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T_softmax)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def gram_loss(self, gram_adv: Tensor, idx: int) -> Tensor:
        """
        Gram MSE loss:对抗图片的 Gram vs 目标图片的 Gram,
        通过 cosine distance 度量纹理/风格相似度。

        gram_adv shape: [D, D] (squeezed from [1, D, D])
        gram_gt stored shape: [1, D, D]
        """
        gram_gt = self.ground_truth_gram[idx].squeeze(0)  # [D, D]
        # Cosine distance on flattened Gram matrices:
        #  - 1.0 = identical style
        #  - lower = more different textures
        adv_flat = gram_adv.view(-1)
        gt_flat = gram_gt.view(-1)
        cosine_sim = torch.nn.functional.cosine_similarity(adv_flat.unsqueeze(0), gt_flat.unsqueeze(0))
        return 1.0 - cosine_sim  # minimize = maximize style similarity

    def get_cluster_center(self, embedding_img, device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device,
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return torch.tensor(0.0, device=src_dis.device)
        sim_op = torch.sum(T * sim, dim=(0, 1))
        return torch.sum(sim_op)

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for _ in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        return torch.outer(r, c) * K
