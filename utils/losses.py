import torch
import torch.nn.functional as F


def loss_l2(feature_s, feature_t):
    """
    Distillation between the student and teacher output
        feature_s: features of student aligned to the size of features of teacher
        feature_t: features of teacher
    """

    loss_type = torch.nn.MSELoss()
    loss = 0.0
    for i in range(len(feature_s)):
        loss_i = loss_type(feature_s[i], feature_t[i])
        loss += loss_i

    return loss


def loss_distil(feature_s, feature_t):
    """
    Distillation between the student and teacher output
        feature_s: features of student aligned to the size of features of teacher
        feature_t: features of teacher
    """

    loss_type = torch.nn.CosineSimilarity()
    # loss_type = nn.L1Loss()
    loss = 0.0
    for i in range(len(feature_s)):
        # print(feature_s[i].shape)
        # print(feature_t[i].shape)
        loss_i = torch.mean(1 - loss_type(
            feature_s[i].view(feature_s[i].shape[0], -1),
            feature_t[i].view(feature_t[i].shape[0], -1)))
        # print(loss_i)
        # self.weight_map[i] = 2 - loss_i
        # print(self.weight_map[i])
        loss += loss_i

    return loss


def loss_distil_p(feature_s, feature_t):
    """
    Distillation between the student and teacher output
        feature_s: features of student aligned to the size of features of teacher
        feature_t: features of teacher
    """

    loss_type = torch.nn.CosineSimilarity()
    # loss_type = nn.L1Loss()
    loss = 0.0
    for i in range(len(feature_s)):
        # print(feature_s[i].shape)
        # print(feature_t[i].shape)
        cos = 1 - loss_type(feature_s[i], feature_t[i])
        # cos = torch.unsqueeze(1 - cos)
        # cos = F.normalize(cos, p=2, dim=(2, 3))
        loss_i = torch.mean(cos)
        loss += loss_i

    return loss


def loss_distil_pixel(feature_s, feature_t):
    loss = 0.0
    # loss_type = torch.nn.L1Loss()
    loss_type = torch.nn.MSELoss()
    # pdist = torch.nn.PairwiseDistance(p=2)
    pool_s = torch.nn.ModuleList([
        torch.nn.AvgPool2d(4, stride=4),
        torch.nn.AvgPool2d(2, stride=2)
    ])

    pool_t = torch.nn.ModuleList([
        torch.nn.AvgPool2d(4, stride=4),
        torch.nn.AvgPool2d(2, stride=2)
    ])

    # pool_s = torch.nn.AvgPool2d(2, stride=2)
    # pool_t = torch.nn.AvgPool2d(2, stride=2)

    for i in range(len(feature_s)):

        f_s = feature_s[i]
        f_t = feature_t[i]

        # if i==0:
        #     # continue
        #     loss_tmp = 0.0
        #
        #     print(f_s.shape)
        #
        #     # f_s = pool_s[i](feature_s[i])
        #     # f_t = pool_t[i](feature_t[i])
        #
        #     B, C, H, W = f_s.shape
        #     # print(f_s.shape)
        #     for m in range(H // 32):
        #         for n in range(W // 32):
        #             print(m)
        #             print(n)
        #             f_s_patch = f_s[:, :, m * 32:(m + 1) * 32, n * 32:(n + 1) * 32]
        #             # print(f_s_patch.shape)
        #             f_t_patch = f_t[:, :, m * 32:(m + 1) * 32, n * 32:(n + 1) * 32]
        #
        #             cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s_patch, f_t_patch)
        #             loss_tmp += loss_type(cos_sim_s, cos_sim_t)
        #
        #     # print(H // 32)
        #     loss_tmp = loss_tmp / ((H // 32) * (W // 32))
        #     # print(loss_tmp)
        #
        #     loss += loss_tmp
        #     continue
        #
        #     # loss_tmp = loss_tmp * 0.5
        #     # # patch
        #     # f_s = pool_s[i](feature_s[i])
        #     # f_t = pool_t[i](feature_t[i])
        #     #
        #     # cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s, f_t)
        #     #
        #     # # print(cos_sim_s)
        #     # # print(cos_sim_t)
        #     # loss_tmp += loss_type(cos_sim_s, cos_sim_t) * 0.5
        #     # loss += loss_tmp
        #     # continue

        if i == 0 or i == 1:
            f_s = pool_s[i](f_s)
            f_t = pool_t[i](f_t)

        cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s, f_t)

        loss += loss_type(cos_sim_s, cos_sim_t)

    return loss


def calculate_pixel_similarity(f_s, f_t):
    B, C, H, W = f_s.shape
    f_s = f_s.contiguous().view(B, C, -1)
    f_s = f_s.transpose(1, 2)
    # f_s = torch.mean(f_s.transpose(1, 2), dim=-1, keepdim=True)
    cos_sim_s = torch.empty(f_s.shape[0], f_s.shape[1], f_s.shape[1]).to(f_s.device)
    # pair_sim_s = torch.empty(f_s.shape[0], f_s.shape[1], f_s.shape[1])
    for batch in range(f_s.shape[0]):
        cos_sim_s[batch] = F.cosine_similarity(f_s.unsqueeze(2)[batch:batch + 1, :, :, :],
                                               f_s.unsqueeze(1)[batch:batch + 1, :, :, :], dim=-1)

    B, C, H, W = f_t.shape
    f_t = f_t.contiguous().view(B, C, -1)
    f_t = f_t.transpose(1, 2)
    # f_t = torch.mean(f_t.transpose(1, 2), dim=-1, keepdim=True)
    cos_sim_t = torch.empty(f_t.shape[0], f_t.shape[1], f_t.shape[1]).to(f_s.device)
    for batch in range(f_t.shape[0]):
        cos_sim_t[batch] = F.cosine_similarity(f_t.unsqueeze(2)[batch:batch + 1, :, :, :],
                                               f_t.unsqueeze(1)[batch:batch + 1, :, :, :], dim=-1)

    return cos_sim_s, cos_sim_t


def calculate_pixel_similarity_f(f):
    B, C, H, W = f.shape
    f = f.contiguous().view(B, C, -1)
    f = f.transpose(1, 2)
    cos_sim = torch.empty(f.shape[0], f.shape[1], f.shape[1]).to(f.device)
    # pair_sim_s = torch.empty(f_s.shape[0], f_s.shape[1], f_s.shape[1])
    for batch in range(f.shape[0]):
        cos_sim[batch] = F.cosine_similarity(f.unsqueeze(2)[batch:batch + 1, :, :, :],
                                               f.unsqueeze(1)[batch:batch + 1, :, :, :], dim=-1)

    return cos_sim


def calculate_pixel_similarity_f_2(f_s, f_t):
    B, C, H, W = f_s.shape
    f_s = f_s.contiguous().view(B, C, -1)
    f_s = f_s.transpose(1, 2)
    f_t = f_t.contiguous().view(B, C, -1)
    f_t = f_t.transpose(1, 2)
    cos_sim = torch.empty(f_s.shape[0], f_s.shape[1], f_s.shape[1]).to(f_s.device)
    # pair_sim_s = torch.empty(f_s.shape[0], f_s.shape[1], f_s.shape[1])
    for batch in range(f_s.shape[0]):
        cos_sim[batch] = F.cosine_similarity(f_s.unsqueeze(2)[batch:batch + 1, :, :, :],
                                               f_t.unsqueeze(1)[batch:batch + 1, :, :, :], dim=-1)

    return cos_sim


# def loss_pixel_attention(feature_s, feature_t):
#     loss = 0.0
#     loss_type = torch.nn.L1Loss()
#
#     for i in range(len(feature_s)):
#
#         f_s = feature_s[i]
#         f_t = feature_t[i]
#
#         if i == 0 or i == 1 or i==2:
#             # continue
#             loss_tmp = 0.0
#
#             # print(f_s.shape)
#
#             # f_s = self.pool_s[i](feature_s[i])
#             # f_t = self.pool_t[i](feature_t[i])
#
#             B, C, H, W = f_s.shape
#             # print(f_s.shape)
#             for m in range(H // 16):
#                 for n in range(W // 16):
#                     f_s_patch = f_s[:, :, m * 16:(m + 1) * 16, n * 16:(n + 1) * 16]
#                     # print(f_s_patch.shape)
#                     f_t_patch = f_t[:, :, m * 16:(m + 1) * 16, n * 16:(n + 1) * 16]
#
#                     # cos_sim_st = self.calculate_pixel_similarity_f_2(f_s_patch, f_t_patch)
#                     # cos_sim_t = self.calculate_pixel_similarity_f(f_t_patch)
#                     #
#                     # loss_tmp += torch.mean((1 - cos_sim_st) * cos_sim_t)
#
#                     cos_sim_st = calculate_pixel_similarity_f_2(f_s_patch, f_t_patch)
#                     cos_sim_t = calculate_pixel_similarity_f(f_t_patch)
#                     cos_sim_s = calculate_pixel_similarity_f(f_s_patch)
#
#                     loss_tmp += loss_type(cos_sim_s, cos_sim_t)
#                     with torch.no_grad():
#                         loss_w = 1 - torch.abs(cos_sim_t - cos_sim_s)
#                         # print(loss_w)
#                     # loss_tmp += torch.mean((1 - cos_sim_st) * cos_sim_t * loss_w)
#                     loss_tmp += torch.mean(torch.abs(cos_sim_st - cos_sim_t) * loss_w)
#
#             # print(H // 32)
#             loss_tmp = loss_tmp / ((H // 16) * (W // 16))
#             # print(loss_tmp)
#
#             loss += F.relu(self.weight[i]) * loss_tmp
#             # loss += loss_tmp
#
#             continue
#
#         # cos_sim_st = self.calculate_pixel_similarity_f_2(f_s, f_t)
#         # cos_sim_t = self.calculate_pixel_similarity_f(f_t)
#         #
#         # loss += torch.mean((1-cos_sim_st) * cos_sim_t)
#
#         cos_sim_st = calculate_pixel_similarity_f_2(f_s, f_t)
#         cos_sim_t = calculate_pixel_similarity_f(f_t)
#         cos_sim_s = calculate_pixel_similarity_f(f_s)
#
#         loss_tmp = loss_type(cos_sim_s, cos_sim_t)
#         with torch.no_grad():
#             loss_w = 1 - torch.abs(cos_sim_t - cos_sim_s)
#             # print(loss_w)
#         loss_tmp += torch.mean(torch.abs(cos_sim_st - cos_sim_t) * loss_w)
#         # loss += loss_tmp
#         loss += F.relu(weight[i]) * loss_tmp
#
#     return loss