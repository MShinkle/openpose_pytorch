        
import torch
# from scipy.ndimage.filters import gaussian_filter

map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
            [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
            [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38],
            [45, 46]]
limbseq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18], [3, 17], [6, 18]]

def tensor_delete(tensor, indices, axis):
    good_idxs = [idx for idx in range(tensor.shape[axis]) if idx not in torch.tensor(indices)]
    if len(good_idxs) > 0:
        return torch.index_select(tensor, axis, torch.tensor(good_idxs))
    else:
        return []

def get_candidates_subsets(heatmaps, pafs, thresh_1 = 0.1, thresh_2 = 0.05):
    heatmaps = heatmaps.moveaxis(1,-1)
    pafs = pafs.moveaxis(1,-1)
    candidates, subsets = [], []
    for heatmap, paf in zip(heatmaps, pafs):
        all_peaks = []
        num_peaks = 0
        for part in range(18):
            map_orig = heatmap[:, :, part]
            # map_filt = gaussian_filter(map_orig, sigma=3)
            map_filt = map_orig
            
            map_L = torch.zeros_like(map_filt)
            map_T = torch.zeros_like(map_filt)
            map_R = torch.zeros_like(map_filt)
            map_B = torch.zeros_like(map_filt)
            map_L[1:, :] = map_filt[:-1, :]
            map_T[:, 1:] = map_filt[:, :-1]
            map_R[:-1, :] = map_filt[1:, :]
            map_B[:, :-1] = map_filt[:, 1:]
            
            peaks_binary = ((map_filt >= map_L) * (map_filt >= map_T) * (map_filt >= map_R) * (map_filt >= map_B) * (map_filt > thresh_1))
            peaks = list(zip(torch.nonzero(peaks_binary).T[1], torch.nonzero(peaks_binary).T[0]))
            peaks_ids = range(num_peaks, num_peaks + len(peaks))
            peaks_with_scores = [peak + (map_orig[peak[1], peak[0]].item(),) for peak in peaks]
            peaks_with_scores_and_ids = [peaks_with_scores[i] + (peaks_ids[i],) \
                                            for i in range(len(peaks_ids))]
            all_peaks.append(peaks_with_scores_and_ids)
            num_peaks += len(peaks)
        
        all_connections = []
        spl_k = []
        mid_n = 10
        
        for k in range(len(map_idx)):
            score_mid = paf[:, :, [x - 19 for x in map_idx[k]]]
            candidate_A = all_peaks[limbseq[k][0] - 1]
            candidate_B = all_peaks[limbseq[k][1] - 1]
            n_A = len(candidate_A)
            n_B = len(candidate_B)
            index_A, index_B = limbseq[k]
            if n_A != 0 and n_B != 0:
                connection_candidates = []
                for i in range(n_A):
                    for j in range(n_B):
                        v = torch.subtract(torch.tensor(candidate_B[j][:2]), torch.tensor(candidate_A[i][:2]))
                        n = torch.sqrt(v[0] * v[0] + v[1] * v[1])
                        v = v / n
                        
                        ab = torch.round(torch.stack([torch.linspace(candidate_A[i][0], candidate_B[j][0], mid_n),
                                        torch.linspace(candidate_A[i][1], candidate_B[j][1], mid_n)])).to(int)
                        vx, vy = score_mid[ab[1], ab[0]].T
                        score_midpoints = (vx * v[0]) + (vy * v[1])
                        score_with_dist_prior = (sum(score_midpoints) / len(score_midpoints) + min(0.5 * heatmaps.shape[1] / n - 1, 0)).item()
                        criterion_1 = len(torch.nonzero((score_midpoints > thresh_2)).T[0]) > 0.8 * len(score_midpoints)
                        criterion_2 = score_with_dist_prior > 0
                        if criterion_1 and criterion_2:
                            connection_candidate = [i, j, score_with_dist_prior, score_with_dist_prior + candidate_A[i][2] + candidate_B[j][2]]
                            connection_candidates.append(connection_candidate)
                connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)
                connection = torch.zeros((0, 5))
                for candidate in connection_candidates:
                    i, j, s = candidate[0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = torch.vstack([connection, torch.tensor([candidate_A[i][3], candidate_B[j][3], s, i, j])])
                        if len(connection) >= min(n_A, n_B):
                            break
                all_connections.append(connection)
            else:
                spl_k.append(k)
                all_connections.append([])

        candidate = torch.tensor([item for sublist in all_peaks for item in sublist])
        candidates.append(candidate)
        subset = torch.ones((0, 20)) * -1
# 
        for k in range(len(map_idx)):
            if k not in spl_k:
                part_As = all_connections[k][:, 0]
                part_Bs = all_connections[k][:, 1]
                index_A, index_B = torch.tensor(limbseq[k]) - 1
                for i in range(len(all_connections[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][index_A] == part_As[i] or subset[j][index_B] == part_Bs[i]:
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_B] != part_Bs[i]:
                            subset[j][index_B] = part_Bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_Bs[i].to(int), 2] + all_connections[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).type(torch.int) + (subset[j2] >= 0).type(torch.int))[:-2]
                        if len(torch.nonzero(membership == 2).T[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += all_connections[k][i][2]
                            subset = tensor_delete(subset, j2, 0)
                        else:
                            subset[j1][index_B] = part_Bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_Bs[i].type(torch.int), 2] + all_connections[k][i][2]
                    elif not found and k < 17:
                        row = torch.ones(20) * -1
                        row[index_A] = part_As[i]
                        row[index_B] = part_Bs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[all_connections[k][i, :2].to(int), 2]) + all_connections[k][i][2]
                        subset = torch.vstack([subset, row])
        # 
        del_idx = []
        # 
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                del_idx.append(i)
        subset = tensor_delete(subset, del_idx, axis=0)
        subsets.append(subset)
    return candidates, subsets

def get_keypoints(candidates, subsets):
    k = len(subsets)
    keypoints = torch.zeros((k, 18, 3), dtype=torch.int)
    for i in range(k):
        for j in range(18):
            index = subsets[i][j].to(int)
            if index != -1:
                x, y = candidates[index][:2].to(int)
                keypoints[i][j] = torch.tensor([x, y, 1])
    return keypoints