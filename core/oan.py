import torch
import torch.nn as nn
from loss import batch_episym
import numpy as np
import cv2
import pdb

class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):
    def __init__(self, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points1),
                nn.ReLU(),
                nn.Conv2d(points1, points2, kernel_size=1),
                nn.BatchNorm2d(points2),
                nn.ReLU(),
                nn.Conv2d(points2, points1, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)# b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):
        #x_up: b*c*n*1
        #x_down: b*c*k*1
        embed = self.conv(x_up)# b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)# b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out

def knn(x, k, ignore_self=False, normalized=True):
    """ find feature space knn neighbor of x 
    Input:
        - x:       [bs, num_corr, num_channels],  input features
        - k:       
        - ignore_self:  True/False, return knn include self or not.
        - normalized:   True/False, if the feature x normalized.
    Output:
        - idx:     [bs, num_corr, k], the indices of knn neighbors
    """
    inner = 2 * torch.matmul(x, x.transpose(2, 1))
    if normalized:
        pairwise_distance = 2 - inner
    else:
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = xx - inner + xx.transpose(2, 1)

    if ignore_self is False:
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    else:
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx

def spatial_knn(x, k, ignore_self=False):
    pairwise_distance = ((x[:,None,:,:] - x[:,:,None,:])**2).sum(-1)
    if ignore_self is False:
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    else:
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


class OANBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters, obj_geod_th):
        nn.Module.__init__(self)
        channels = net_channels
        self.sigma = torch.Tensor([1.0]).float().cuda()
        # self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=False)
        self.obj_geod_th = obj_geod_th
        self.layer_num = depth
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PointCN(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(2*channels, channels))
        for _ in range(self.layer_num//2-1):
            self.l1_2.append(PointCN(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs, ys=None):
        #data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2( torch.cat([x1_1,x_up], dim=1))

        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        if data.shape[1] == 4:
            weights = torch.relu(torch.tanh(logits))
            e_hat = weighted_8points(xs, weights)

            x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
            e_hat_norm = e_hat
            residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

            normed_corr_features = nn.functional.normalize(out.squeeze(-1).permute(0,2,1), dim=-1)
            M = torch.matmul(normed_corr_features, normed_corr_features.permute(0, 2, 1))
            M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
            return logits, e_hat, residual, M, {}

        normed_corr_features = nn.functional.normalize(out.squeeze(-1).permute(0,2,1), dim=-1)
        M = torch.matmul(normed_corr_features, normed_corr_features.permute(0, 2, 1))
        M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
            
        #################################
        # Step 2.1: estimate initial confidence by MLP, find highly confident and well-distributed points as seeds.
        #################################
        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        seeds = torch.argsort(logits, dim=1, descending=True)[:, 0:2]
        # seeds = torch.argsort(ys[:,:,0], dim=1, descending=False)[:, 0:int(num_pts * 0.01)]
        
        #################################
        # Step 3 & 4: calculate transformation matrix for each seed, and find the best hypothesis.
        #################################
        final_labels, stats = self.cal_seed_trans(seeds, normed_corr_features, xs[:,0,:,:2], xs[:,0,:,2:4], logits, ys)

        #################################
        # Step 5: post refinement
        #################################
        e_hat = weighted_8points(xs, final_labels)
        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual, M, stats

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input: 
            - M:      [bs, num_corr, num_corr] the compatibility matrix 
            - method: select different method for calculating the learding eigenvector.
        Output: 
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(10):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        else:
            exit(-1)

    def cal_seed_trans(self, seeds, corr_features, src_keypts, tgt_keypts, logits, ys=None):
        """
        Calculate the transformation for each seeding correspondences.
        Input: 
            - seeds:         [bs, num_seeds]              the index to the seeding correspondence
            - corr_features: [bs, num_corr, num_channels]
            - src_keypts:    [bs, num_corr, 2]
            - tgt_keypts:    [bs, num_corr, 2]
        """
        stats = {}
        bs, num_corr, num_channels = corr_features.shape[0], corr_features.shape[1], corr_features.shape[2]
        num_seeds = seeds.shape[-1]
        k = min(200, num_corr - 1)
        # knn_idx = spatial_knn(src_keypts, k=k, ignore_self=True)
        knn_idx = knn(corr_features, k=k, ignore_self=True, normalized=True)  # [bs, num_corr, k]
        knn_idx = knn_idx.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, k)) 
        # stats
        if ys is not None:
            knn_ys = ys.gather(dim=1, index=knn_idx.view([1, -1, 1])).view(knn_idx.shape)
            knn_labels = (knn_ys < self.obj_geod_th).float()
            knn_inlier_ratio = knn_labels.mean(-1)
            stats['inlier_ratio_mean'] = knn_inlier_ratio.mean().item()
            stats['inlier_ratio_max'] = knn_inlier_ratio.max().item()

        #################################
        # construct the feature consistency matrix of each correspondence subset.
        #################################
        knn_features = corr_features.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, num_channels)).view([bs, -1, k, num_channels])  # [bs, num_seeds, k, num_channels]
        knn_M = torch.matmul(knn_features, knn_features.permute(0, 1, 3, 2))
        knn_M = torch.clamp(1 - (1 - knn_M) / self.sigma ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        feature_knn_M = knn_M

        #################################
        # construct the spatial consistency matrix of each correspondence subset.
        #################################
        # src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])  # [bs, num_seeds, k, 3]
        # tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])
        # knn_M = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5 - ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        # knn_M = torch.clamp(1 - knn_M ** 2/ self.sigma_spat ** 2, min=0)
        # knn_M = knn_M.view([-1, k, k])
        # spatial_knn_M = knn_M

        #################################
        # Power iteratation to get the inlier probability
        #################################
        total_knn_M = feature_knn_M
        total_knn_M[:, torch.arange(total_knn_M.shape[1]), torch.arange(total_knn_M.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(total_knn_M, method='power')
        total_weight = total_weight.view([bs, -1, k])
        # total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel 
        #################################
        # total_weight = knn_labels
        # knn_logits = logits.gather(dim=1, index=knn_idx.view([1, -1])).view(knn_idx.shape)
        # total_weight = torch.relu(torch.tanh(knn_logits))

        src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 2)).view([bs, -1, k, 2])  # [bs, num_seeds, k, 2]
        tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 2)).view([bs, -1, k, 2])  # [bs, num_seeds, k, 2]
        src_knn, tgt_knn = src_knn.view([-1, k, 2]), tgt_knn.view([-1, k, 2])

        local_xs = torch.cat([src_knn, tgt_knn], dim=-1)[:, None, :, :]
        seedwise_trans = weighted_8points(local_xs, total_weight) # [bs * num_seeds, 9]

        # seedwise_trans = np.eye(3)[None, :, :].repeat(num_seeds, axis=0)
        # src_knn_cpu = src_knn.detach().cpu().numpy()
        # tgt_knn_cpu = tgt_knn.detach().cpu().numpy()
        # import time
        # s_time = time.time()
        # for ii in range(num_seeds):
        #     E, mask_new = cv2.findEssentialMat(src_knn_cpu[ii], tgt_knn_cpu[ii], method=cv2.RANSAC, prob=0.999, threshold=0.001)
        #     seedwise_trans[ii] = E
        # e_time = time.time()
        # print('RANSAC 0.999', e_time - s_time)
        # seedwise_trans = torch.from_numpy(seedwise_trans).float().to(src_knn.device)
        # seedwise_trans = seedwise_trans.view([-1, 9])
        
        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        e_hat_norm = seedwise_trans
        ind = torch.arange(bs)[None].repeat(num_seeds, 1).T.reshape(-1, 1).squeeze()
        residual = batch_episym(src_keypts[ind,:,:], tgt_keypts[ind,:,:], e_hat_norm)#.reshape(batch_size, 1, num_pts, 1)
        seedwise_fitness = torch.mean((residual < self.obj_geod_th).float(), dim=-1) 
        seedwise_labels = (residual < self.obj_geod_th)


        seedwise_fitness = seedwise_fitness.reshape(bs, num_seeds)
        seedwise_trans = seedwise_trans.reshape(bs, num_seeds, -1)
        residual = residual.reshape(bs, num_seeds, -1)
        batch_best_guess = seedwise_fitness.argmax(dim=1)  
        final_trans = seedwise_trans.gather(dim=1, index=batch_best_guess[:, None, None].expand(-1, -1, 9)).squeeze(1)
        final_res = residual.gather(dim=1, index=batch_best_guess[:, None, None].expand(-1, -1, num_corr)).squeeze(1)
        final_labels = (final_res < self.obj_geod_th).float()

        if ys is not None:
            stats['select_inlier_ratio'] = knn_inlier_ratio[0, batch_best_guess[0]].item()
        return final_labels, stats

class OANet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)
        self.weights_init = OANBlock(config.net_channels, 4+self.side_channel, depth_each_stage, config.clusters, config.obj_geod_th)
        self.weights_iter = [OANBlock(config.net_channels, 6+self.side_channel, depth_each_stage, config.clusters, config.obj_geod_th) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        #data: b*1*n*c
        input = data['xs'].transpose(1,3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat, Ms = [], [], []
        logits, e_hat, residual, M, stats = self.weights_init(input, data['xs'], data['ys'])
        res_logits.append(logits), res_e_hat.append(e_hat), Ms.append(M)
        for i in range(self.iter_num):
            logits, e_hat, residual, M, stats = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1),
                data['xs'], data['ys'])
            res_logits.append(logits), res_e_hat.append(e_hat), Ms.append(M)
        return res_logits, res_e_hat, Ms, stats


        
def batch_symeig_old(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    device = X.device
    X = X.cpu()
    b, d, _ = X.size()
    ee, vv = torch.symeig(X, True)
    return vv.to(device)

def weighted_8points(x_in, weights, weight_threshold=0):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    # weights = torch.relu(torch.tanh(logits))
    weights[weights < weight_threshold] = 0
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

