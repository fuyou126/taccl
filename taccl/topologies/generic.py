# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from .topology import NodeTopology
from functools import reduce
import operator

def validate_and_modify_topo(topo_json, check_links=True):
    assert "name" in topo_json, "Provide a name in the topo file"
    assert "gpus_per_node" in topo_json
    assert "alpha" in topo_json
    devices = topo_json["gpus_per_node"]
    assert devices > 0
    if check_links:
        assert "links" in topo_json
        assert "invbws" in topo_json
        assert "betas" in topo_json
        assert "node_invbws_list" not in topo_json
        assert "node_betas_list" not in topo_json
        assert len(topo_json["links"]) == devices
        assert len(topo_json["betas"]) == devices
        assert len(topo_json["invbws"]) == devices
        for l in topo_json["links"]:
            assert isinstance(l, list)
            assert len(l) == devices
        for l in topo_json["invbws"]:
            assert isinstance(l, list)
            assert len(l) == devices
        for l in topo_json["betas"]:
            assert isinstance(l, list)
            assert len(l) == devices
    else:
        assert "links" not in topo_json
        assert "invbws" not in topo_json
        assert "node_invbws_list" in topo_json
        assert "node_betas_list" in topo_json
    if ("nics_per_node" in topo_json):
        assert "remote_alpha" in topo_json
        assert "remote_beta" in topo_json
        assert "remote_invbw" in topo_json
    else:
        topo_json["nics_per_node"] = -1
        topo_json["remote_alpha"] = -1
        topo_json["remote_beta"] = -1
        topo_json["remote_invbw"] = -1
    return topo_json

def custom(topo_file):
    f = open(topo_file, "r")
    topo_json = json.load(f)
    topo_json = validate_and_modify_topo(topo_json, check_links=True)
    gpus_per_node = topo_json["gpus_per_node"]
    alpha = topo_json["alpha"]
    links = topo_json["links"]
    invbws = topo_json["invbws"]
    betas = topo_json["betas"]
    nics_per_node = topo_json["nics_per_node"]
    remote_invbw = topo_json["remote_invbw"]
    remote_alpha = topo_json["remote_alpha"]
    remote_beta = topo_json["remote_beta"]
    name = topo_json["name"]
    return NodeTopology(f'Custom-{name}-(n={gpus_per_node})', links, alpha, betas, invbws, nics_per_node, remote_invbw, remote_alpha, remote_beta)


def hub_and_spoke(topo_file):
    print("topo_file:", topo_file)
    f = open(topo_file, "r")
    topo_json = json.load(f)
    gpus_per_node = topo_json["gpus_per_node"]
    assert len(topo_json["node_invbws_list"]) == 1
    node_invbw = topo_json["node_invbws_list"][0]
    assert len(topo_json["node_betas_list"]) == 1
    node_beta = topo_json["node_betas_list"][0]
    alpha = topo_json["alpha"]
    links = [[0 if x==y else 1 for y in range(gpus_per_node)] for x in range(gpus_per_node)]
    betas = [[0 if x==y else node_beta for y in range(gpus_per_node)] for x in range(gpus_per_node)]
    invbws = [[0 if x==y else node_invbw for y in range(gpus_per_node)] for x in range(gpus_per_node)]
    nics_per_node = topo_json["nics_per_node"]
    remote_invbw = topo_json["remote_invbw"]
    remote_alpha = topo_json["remote_alpha"]
    remote_beta = topo_json["remote_beta"]
    name = topo_json["name"]
    return NodeTopology(f'HubAndSpoke-{name}-(n={gpus_per_node})', links, alpha, betas, invbws, nics_per_node, remote_invbw, remote_alpha, remote_beta)


def dgx2(topo_file):
    print("topo_file:", topo_file)
    print("------------------------testDGX-2--------------------------------")
    f = open(topo_file, "r")
    topo_json = json.load(f)
    assert topo_json["nics_per_node"] == 8
    assert topo_json["gpus_per_node"] == 16
    print("Fixing nics_per_node and gpus_per_node. This will overwrite any values provided")
    topo_json = validate_and_modify_topo(topo_json, check_links=False)
    assert len(topo_json["node_invbws_list"]) == 1
    assert len(topo_json["node_betas_list"]) == 1
    node_invbw = int(topo_json["node_invbws_list"][0])
    node_beta = topo_json["node_betas_list"][0]
    alpha = topo_json["alpha"]
    gpus_per_node = topo_json["gpus_per_node"]
    nics_per_node = topo_json["nics_per_node"]
    remote_invbw = topo_json["remote_invbw"]
    remote_alpha = topo_json["remote_alpha"]
    remote_beta = topo_json["remote_beta"]
    name = topo_json["name"]
    links = [[0 if x==y else 1 for y in range(gpus_per_node)] for x in range(gpus_per_node)]
    betas = [[0 if x==y else node_beta for y in range(gpus_per_node)] for x in range(gpus_per_node)]
    invbws = [[0 if x==y else node_invbw for y in range(gpus_per_node)] for x in range(gpus_per_node)]
    return NodeTopology(f'DGX2-{name}-(n={gpus_per_node})', links, alpha, betas, invbws, nics_per_node, remote_invbw, remote_alpha, remote_beta)


def ndv2(topo_file):
    print("topo_file:", topo_file)
    f = open(topo_file, "r")
    topo_json = json.load(f)
    f.close()
    assert topo_json["nics_per_node"] == 1
    assert topo_json["gpus_per_node"] == 8
    print("Fixing nics_per_node and gpus_per_node. This will overwrite any values provided")
    topo_json = validate_and_modify_topo(topo_json, check_links=False)
    assert len(topo_json["node_invbws_list"]) == 2

    # Link connection matrix
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]

    alpha = topo_json["alpha"]

    # NVLink beta for each link
    beta_m1 = topo_json["node_betas_list"][0]
    beta_m2 = topo_json["node_betas_list"][1]
    betas = [
        [0, beta_m1, beta_m2, beta_m2, beta_m1, 0, 0, 0],
        [beta_m1, 0, beta_m2, beta_m1, 0, beta_m2, 0, 0],
        [beta_m2, beta_m2, 0, beta_m1, 0, 0, beta_m1, 0],
        [beta_m2, beta_m1, beta_m1, 0, 0, 0, 0, beta_m2],
        [beta_m1, 0, 0, 0, 0, beta_m1, beta_m2, beta_m2],
        [0, beta_m2, 0, 0, beta_m1, 0, beta_m2, beta_m1],
        [0, 0, beta_m1, 0, beta_m2, beta_m2, 0, beta_m1],
        [0, 0, 0, beta_m2, beta_m2, beta_m1, beta_m1, 0]
    ]

    # NVLink bandwidth for each link
    invbw1 = int(topo_json["node_invbws_list"][0])
    invbw2 = int(topo_json["node_invbws_list"][1])
    invbws = [
        [0, invbw1, invbw2, invbw2, invbw1, 0, 0, 0],
        [invbw1, 0, invbw2, invbw1, 0, invbw2, 0, 0],
        [invbw2, invbw2, 0, invbw1, 0, 0, invbw1, 0],
        [invbw2, invbw1, invbw1, 0, 0, 0, 0, invbw2],
        [invbw1, 0, 0, 0, 0, invbw1, invbw2, invbw2],
        [0, invbw2, 0, 0, invbw1, 0, invbw2, invbw1],
        [0, 0, invbw1, 0, invbw2, invbw2, 0, invbw1],
        [0, 0, 0, invbw2, invbw2, invbw1, invbw1, 0]
    ]
    # Ex. for 1 MB data chunks, the following matrix denotes node invbws
    # invbws = [
    #     [0, 23, 46, 46, 23, 0, 0, 0],
    #     [23, 0, 46, 23, 0, 46, 0, 0],
    #     [46, 46, 0, 23, 0, 0, 23, 0],
    #     [46, 23, 23, 0, 0, 0, 0, 46],
    #     [23, 0, 0, 0, 0, 23, 46, 46],
    #     [0, 46, 0, 0, 23, 0, 46, 23],
    #     [0, 0, 23, 0, 46, 46, 0, 23],
    #     [0, 0, 0, 46, 46, 23, 23, 0]
    # ]
    nics_per_node = topo_json["nics_per_node"]
    remote_invbw = topo_json["remote_invbw"]
    remote_alpha = topo_json["remote_alpha"]
    remote_beta = topo_json["remote_beta"]
    name = topo_json["name"]

    return NodeTopology(f'NDv2-{name}', links, alpha, betas, invbws, nics_per_node, remote_invbw, remote_alpha, remote_beta)


# have to input dims and wrap instead of links
# if both of betas and invbws are [], it means all of invbws are same
def mesh(topo_file):
    with open(topo_file, "r") as f:
        topo_json = json.load(f)
    dims = topo_json["dims"]        # 例如 [nx, ny] 或 [nx, ny, nz]
    wrap = topo_json["wrap"]        # 每一维是否是环状
    assert len(dims) == len(wrap), "dims 和 wrap 的长度必须一致"
    gpus_per_node = topo_json["gpus_per_node"]
    alpha = topo_json["alpha"]
    # 计算总节点数，并验证与 gpus_per_node 是否一致
    total_nodes = reduce(operator.mul, dims, 1)
    assert total_nodes == gpus_per_node, "gpus_per_node 与 dims 的乘积不匹配"
    # 初始化邻接矩阵：total_nodes x total_nodes 全部置0
    links = [[0 for _ in range(total_nodes)] for _ in range(total_nodes)]
    # 辅助函数：将多维坐标转换为一维索引（按照每一维大小计算权重，采用逆序累乘）

    def coord_to_index(coord):
        idx = 0
        multiplier = 1
        # 采用从最后一维到第一维的顺序计算
        for c, dim in zip(reversed(coord), reversed(dims)):
            idx += c * multiplier
            multiplier *= dim
        return idx

    # 辅助函数：将一维索引转换为多维坐标
    def index_to_coord(idx):
        coord = []
        for dim in reversed(dims):
            coord.append(idx % dim)
            idx //= dim
        return list(reversed(coord))

    # 遍历每个节点，根据每个维度查找相邻节点
    for idx in range(total_nodes):
        coord = index_to_coord(idx)
        # 对于每一维，尝试找前向和后向的邻居
        for d in range(len(dims)):
            # 负方向邻居
            new_coord = coord.copy()
            new_coord[d] -= 1
            if new_coord[d] < 0:
                if wrap[d]:
                    new_coord[d] = dims[d] - 1  # 环状连接：从另一侧接入
                else:
                    new_coord = None  # Mesh边界无连接
            if new_coord is not None:
                neighbor_index = coord_to_index(new_coord)
                links[idx][neighbor_index] = 1
                links[neighbor_index][idx] = 1  # 保证对称

            # 正方向邻居
            new_coord = coord.copy()
            new_coord[d] += 1
            if new_coord[d] >= dims[d]:
                if wrap[d]:
                    new_coord[d] = 0  # 环状连接：回到起始位置
                else:
                    new_coord = None
            if new_coord is not None:
                neighbor_index = coord_to_index(new_coord)
                links[idx][neighbor_index] = 1
                links[neighbor_index][idx] = 1

    # 根据 links 生成 betas 和 invbws（若原始配置中为空）
    betas = topo_json["betas"]
    if len(betas) == 0:
        # 生成与 links 同尺寸的矩阵，每个连接权重乘以 5
        betas = [[v * 5 for v in row] for row in links]
    invbws = topo_json["invbws"]
    if len(invbws) == 0:
        # 同理，生成每个连接带宽逆值乘以 15
        invbws = [[v * 15 for v in row] for row in links]

    nics_per_node = topo_json["nics_per_node"]
    remote_invbw = topo_json["remote_invbw"]
    remote_alpha = topo_json["remote_alpha"]
    remote_beta = topo_json["remote_beta"]
    name = topo_json["name"]

    return NodeTopology(
        f'Mesh-{name}-(n={gpus_per_node})',
        links,
        alpha,
        betas,
        invbws,
        nics_per_node,
        remote_invbw,
        remote_alpha,
        remote_beta,
        dims,
        wrap
    )