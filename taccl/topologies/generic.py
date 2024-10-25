# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from .topology import NodeTopology

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
        topo_json["remote_invbws"] = -1
    return topo_json

def custom(topo_file):
    topo_json = json.load(topo_file)
    topo_json = validate_and_modify_topo(topo_json, check_links=True)
    gpus_per_node = topo_json["gpus_per_node"]
    links = topo_json["links"]
    invbws = topo_json["invbws"]
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
    topo_json["nics_per_node"] = 8
    topo_json["gpus_per_node"] == 16
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
    topo_json["nics_per_node"] = 1
    topo_json["gpus_per_node"] == 8
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


def torus2d(topo_file):
    # 读取 JSON 配置文件
    with open(topo_file, "r") as f:
        topo_json = json.load(f)

    print("============================2D Torus 测试===================================")
    # 设置设备数和拓扑名称
    gpus_per_node = topo_json["gpus_per_node"]
    alpha = topo_json["alpha"]
    node_beta = topo_json["node_betas_list"][0]
    node_invbw = topo_json["node_invbws_list"][0]
    nics_per_node = topo_json.get("nics_per_node", -1)
    remote_invbw = topo_json.get("remote_invbw", -1)
    remote_alpha = topo_json.get("remote_alpha", -1)
    remote_beta = topo_json.get("remote_beta", -1)
    name = topo_json["name"]

    # 假设 gpus_per_node 是平方数，以形成 2D 环形网格
    grid_size = int(gpus_per_node ** 0.5)
    assert grid_size * grid_size == gpus_per_node, "gpus_per_node 必须是完全平方数"

    # 构建 2D 环形网格的连接矩阵 links、beta 矩阵和 invbw 矩阵
    links = [[0 for _ in range(gpus_per_node)] for _ in range(gpus_per_node)]
    betas = [[0 for _ in range(gpus_per_node)] for _ in range(gpus_per_node)]
    invbws = [[0 for _ in range(gpus_per_node)] for _ in range(gpus_per_node)]

    for i in range(grid_size):
        for j in range(grid_size):
            current = i * grid_size + j
            neighbors = [
                ((i - 1) % grid_size) * grid_size + j,  # 上边界连接下侧
                ((i + 1) % grid_size) * grid_size + j,  # 下边界连接上侧
                i * grid_size + (j - 1) % grid_size,  # 左边界连接右侧
                i * grid_size + (j + 1) % grid_size  # 右边界连接左侧
            ]
            for neighbor in neighbors:
                links[current][neighbor] = 1
                betas[current][neighbor] = node_beta
                invbws[current][neighbor] = node_invbw

    return NodeTopology(
        f'Torus-{name}-(n={gpus_per_node})',
        links,
        alpha,
        betas,
        invbws,
        nics_per_node,
        remote_invbw,
        remote_alpha,
        remote_beta
    )

