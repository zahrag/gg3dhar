
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

from pathlib import Path
import numpy as np


def read_MSR(mainpath, input_all_n, class_all, set=0):

    '''
        Read MSRAction3D datasets of 3D actions generate by Kinect type sensor.
        link: https://www.microsoft.com/en-us/research/people/zliu/?from=http%3A%2F%2Fresearch.microsoft.
        com%2Fen-us%2Fum%2Fpeople%2Fzliu%2Factionrecorsrc%2F
        '''

    input_all = []

    nseq = 0
    for na in range(10):
        for ns in range(10):
            for ne in range(3):

                path = get_MSR_filename(na, ns, ne, mainpath, set=set)

                file = Path(path)
                if file.is_file():
                    f = open(path, 'r')
                    lines = f.readlines()  # frames
                    f.close()
                    info = np.array([nseq, ns, na, ne])
                    nseq += 1

                    data_all = np.zeros((len(lines), 80))
                    data_all_n = np.zeros((len(lines), 60))
                    cnt_1_zero = 0
                    for i in range(len(lines)):  # number of frames per sequence
                        var = []
                        k = 0
                        for j in range(len(lines[i])):  # number of characters per frame
                            if lines[i][j] != '\t' and lines[i][j] != '\n':
                                var += lines[i][j]

                            else:
                                if lines[i][j] != '\n':
                                    str = ''.join(var)
                                    data = float(str)
                                    # print(data)
                                    data_all[i, k] = data
                                    del var
                                    del data
                                    var = []
                                    k += 1
                        cnt_2_zero=0
                        for id in range(20):
                            data_all_n[i, 3 * id + 0] = data_all[i, 4 * id + 0]
                            data_all_n[i, 3 * id + 1] = 0.25 * data_all[i, 4 * id + 2]
                            data_all_n[i, 3 * id + 2] = 400 - data_all[i, 4 * id + 1]
                            if data_all_n[i, 3 * id + 0] == 0. and data_all_n[i, 3 * id + 1] == 0. \
                                    and data_all_n[i, 3 * id + 2] == 400.:
                                cnt_2_zero += 1

                        if cnt_2_zero == 20:
                            cnt_1_zero += 1

                    if cnt_1_zero == len(lines):
                        pass
                    else:
                        # Collection of information for each action sequence (sequence, actor, action, event)
                        class_all.append(info)
                        # Collection of raw sensor information 4 values per joint (dim=80)
                        input_all.append(data_all)
                        # Collection of processed 3D information 3 Cartesian coordinate parameters per joint (dim=60)
                        input_all_n.append(data_all_n)

    return input_all_n, class_all


def get_MSR_filename(na, ns, ne, mainpath, set):

    dataset = "MSRAction3DDataset_{:d}".format(set)

    if na + 1 >= 10 and ns + 1 < 10:

        filename = "a{:d}_s0{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    elif na + 1 < 10 and ns + 1 >= 10:

        filename = "a0{:d}_s{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    elif na + 1 >= 10 and ns + 1 >= 10:

        filename = "a{:d}_s{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    else:

        filename = "a0{:d}_s0{:d}_e0{:d}".format(na + 1, ns + 1, ne + 1)

    if set == 2:
        return mainpath + "/" + dataset + "/" + filename + "_2nd.txt"

    else:
        return mainpath + "/" + dataset + "/" + filename + ".txt"


def read_Florence(mainpath, input_all_n, class_all):

    '''
         Florence 3D actions dataset: https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/
        '''

    path = mainpath + "/" + "Florence_3d_actions/Florence_dataset_WorldCoordinates.txt"
    file = Path(path)

    if file.is_file():

        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        # print(lines[0])

        data_all = np.zeros((len(lines), 48))
        for i in range(len(lines)):
            var = []
            k = 0
            for j in range(len(lines[i])):
                if lines[i][j] != ' ':
                    var += lines[i][j]

                else:
                    str = ''.join(var)
                    data = float(str)
                    # print(data)
                    data_all[i, k] = data
                    del var
                    del data
                    var = []
                    k += 1
                if j == len(lines[i]) - 1:
                    str = ''.join(var)
                    data = float(str)
                    # print(data)
                    data_all[i, k] = data
                    del var
                    del data
                    var = []
                    k += 1

        nseq = 1
        prev_nseq = nseq
        mat = np.zeros((1, 45))
        mat2 = np.zeros((1, 3))
        for i in range(np.size(data_all, 0)):
            nseq = data_all[i, 0]
            if nseq == prev_nseq:
                # Data info
                vec = data_all[i, 3:np.size(data_all, 1)]
                vec = np.expand_dims(vec, 0)
                mat = np.vstack((mat, vec))
                # Class info
                vec2 = data_all[i, 0:3]
                vec2 = np.expand_dims(vec2, 0)
                mat2 = np.vstack((mat2, vec2))
            else:
                input_all_n.append(mat[1:np.size(mat, 0), :])
                class_all.append(mat2[1, :])
                prev_nseq = nseq
                del mat
                del mat2
                mat = mat = np.zeros((1, 45))
                mat2 = mat2 = np.zeros((1, 3))

        input_all_n.append(mat[1:np.size(mat, 0), :])
        class_all.append(mat2[1, :])

        # Changing indexes to start from 0
        for nseq in range(len(class_all)):
            class_all[nseq] = class_all[nseq] - 1

    return input_all_n, class_all


def read_UTKinect(mainpath, input_all_n, class_all):

    '''
        UTKinect-Action3D Dataset: http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html
        '''

    input_all = []

    class_all = get_UTKinect_class(class_all)

    for ns in range(10):
        for ne in range(2):
            if ns + 1 < 10:
                path = "{}/UTKinect/joints/joints_s0{:d}_e0{:d}.txt".format(mainpath, ns + 1, ne + 1)
            else:
                path = "{}/UTKinect/joints/joints_s{:d}_e0{:d}.txt".format(mainpath, ns + 1, ne + 1)

            file = Path(path)
            if file.is_file():
                f = open(path, 'r')
                lines = f.readlines()
                f.close()

                data_all = np.zeros((len(lines), 61))
                for i in range(len(lines)):
                    var = []
                    k = 0
                    for j in range(len(lines[i])):
                        if lines[i][j] != ' ':
                            var += lines[i][j]

                        else:
                            if var:
                                str = ''.join(var)
                                data = float(str)
                                # print(data)
                                data_all[i, k] = data
                                del var
                                del data
                                var = []
                                k += 1

                input_all.append(data_all)

    for k in range(len(input_all)):
        data_sub = input_all[k]
        for cnt1 in range(k * 10, (k + 1) * 10):
            data_seq = class_all[cnt1]
            stop = 0
            for cnt2 in range(np.size(data_sub, 0)):
                if data_seq[4] == data_sub[cnt2, 0]:
                    for cnt3 in range(cnt2, np.size(data_sub, 0)):
                        if data_seq[5] == data_sub[cnt3, 0]:
                            input_all_n.append(data_sub[cnt2:cnt3 + 1, 1:61])
                            stop = 1
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break

    return input_all_n, class_all


def get_UTKinect_class(class_all):

    # class_all = np.array([nseq, ns, na, ne, ..., ...])
    # nseq, ns, ne, na

    class_all.append([0, 0, 0, 252, 390])
    class_all.append([0, 0, 1, 572, 686])
    class_all.append([0, 0, 2, 704, 752])
    class_all.append([0, 0, 3, 822, 954])
    class_all.append([0, 0, 4, 1016, 1242])
    class_all.append([0, 0, 5, 1434, 1488])
    class_all.append([0, 0, 6, 1686, 1748])
    class_all.append([0, 0, 7, 1640, 1686])
    class_all.append([0, 0, 8, 1834, 2064])
    class_all.append([0, 0, 9, 2110, 2228])

    class_all.append([0, 1, 0, 154, 194])
    class_all.append([0, 1, 1, 530, 628])
    class_all.append([0, 1, 2, 640, 720])
    class_all.append([0, 1, 3, 1202, 1356])
    class_all.append([0, 1, 4, 1364, 1520])
    class_all.append([0, 1, 5, 2246, 2294])
    class_all.append([0, 1, 6, 2752, 2792])
    class_all.append([0, 1, 7, 2820, 2858])
    class_all.append([0, 1, 8, 2984, 3204])
    class_all.append([0, 1, 9, 3250, 3448])

    class_all.append([1, 0, 0, 266, 368])
    class_all.append([1, 0, 1, 672, 788])
    class_all.append([1, 0, 2, 818, 910])
    class_all.append([1, 0, 3, 1262, 1386])
    class_all.append([1, 0, 4, 1424, 1780])
    class_all.append([1, 0, 5, 2040, 2086])
    class_all.append([1, 0, 6, 2340, 2376])
    class_all.append([1, 0, 7, 2488, 2550])
    class_all.append([1, 0, 8, 2668, 2830])
    class_all.append([1, 0, 9, 3198, 3324])

    class_all.append([1, 1, 0, 40, 208])
    class_all.append([1, 1, 1, 468, 602])
    class_all.append([1, 1, 2, 620, 722])
    class_all.append([1, 1, 3, 894, 1038])
    class_all.append([1, 1, 4, 1340, 1480])
    class_all.append([1, 1, 5, 1966, 2014])
    class_all.append([1, 1, 6, 2194, 2230])
    class_all.append([1, 1, 7, 2314, 2358])
    class_all.append([1, 1, 8, 2408, 2630])
    class_all.append([1, 1, 9, 2690, 2810])

    class_all.append([2, 0, 0, 372, 528])
    class_all.append([2, 0, 1, 734, 862])
    class_all.append([2, 0, 2, 902, 1000])
    class_all.append([2, 0, 3, 1118, 1284])
    class_all.append([2, 0, 4, 1934, 2168])
    class_all.append([2, 0, 5, 3226, 3282])
    class_all.append([2, 0, 6, 3556, 3622])
    class_all.append([2, 0, 7, 3660, 3730])
    class_all.append([2, 0, 8, 3806, 3960])
    class_all.append([2, 0, 9, 4076, 4184])

    class_all.append([2, 1, 0, 122, 254])
    class_all.append([2, 1, 1, 452, 592])
    class_all.append([2, 1, 2, 644, 724])
    class_all.append([2, 1, 3, 848, 1018])
    class_all.append([2, 1, 4, 1078, 1192])
    class_all.append([2, 1, 5, 1638, 1690])
    class_all.append([2, 1, 6, 1866, 1896])
    class_all.append([2, 1, 7, 1928, 2008])
    class_all.append([2, 1, 8, 2054, 2208])
    class_all.append([2, 1, 9, 2324, 2460])

    class_all.append([3, 0, 0, 348, 496])
    class_all.append([3, 0, 1, 788, 864])
    class_all.append([3, 0, 2, 954, 1056])
    class_all.append([3, 0, 3, 1190, 1326])
    class_all.append([3, 0, 4, 1580, 1882])
    class_all.append([3, 0, 5, 2306, 2350])
    class_all.append([3, 0, 6, 2532, 2572])
    class_all.append([3, 0, 7, 2644, 2686])
    class_all.append([3, 0, 8, 2790, 2968])
    class_all.append([3, 0, 9, 3064, 3146])

    class_all.append([3, 1, 0, 420, 546])
    class_all.append([3, 1, 1, 1046, 1144])
    class_all.append([3, 1, 2, 1352, 1414])
    class_all.append([3, 1, 3, 1682, 1820])
    class_all.append([3, 1, 4, 1868, 2122])
    class_all.append([3, 1, 5, 2564, 2608])
    class_all.append([3, 1, 6, 2760, 2792])
    class_all.append([3, 1, 7, 2866, 2910])
    class_all.append([3, 1, 8, 3070, 3260])
    class_all.append([3, 1, 9, 3448, 3622])

    class_all.append([4, 0, 0, 708, 888])
    class_all.append([4, 0, 1, 1140, 1238])
    class_all.append([4, 0, 2, 1294, 1394])
    class_all.append([4, 0, 3, 1482, 1676])
    class_all.append([4, 0, 4, 1736, 2064])
    class_all.append([4, 0, 5, 3104, 3176])
    class_all.append([4, 0, 6, 3596, 3632])
    class_all.append([4, 0, 7, 3706, 3770])
    class_all.append([4, 0, 8, 3946, 4352])
    class_all.append([4, 0, 9, 4522, 4734])

    class_all.append([4, 1, 0, 212, 376])
    class_all.append([4, 1, 1, 646, 756])
    class_all.append([4, 1, 2, 788, 862])
    class_all.append([4, 1, 3, 974, 1180])
    class_all.append([4, 1, 4, 1266, 1540])
    class_all.append([4, 1, 5, 1752, 1828])
    class_all.append([4, 1, 6, 2172, 2230])
    class_all.append([4, 1, 7, 2104, 2156])
    class_all.append([4, 1, 8, 2504, 2784])
    class_all.append([4, 1, 9, 2798, 2900])

    class_all.append([5, 0, 0, 1230, 1366])
    class_all.append([5, 0, 1, 1564, 1644])
    class_all.append([5, 0, 2, 1678, 1758])
    class_all.append([5, 0, 3, 1862, 1948])
    class_all.append([5, 0, 4, 1966, 2098])
    class_all.append([5, 0, 5, 2392, 2414])
    class_all.append([5, 0, 6, 2672, 2698])
    class_all.append([5, 0, 7, 2790, 2824])
    class_all.append([5, 0, 8, 3046, 3216])
    class_all.append([5, 0, 9, 3290, 3444])

    class_all.append([5, 1, 0, 294, 426])
    class_all.append([5, 1, 1, 710, 818])
    class_all.append([5, 1, 2, 856, 956])
    class_all.append([5, 1, 3, 1088, 1174])
    class_all.append([5, 1, 4, 2032, 2202])
    class_all.append([5, 1, 5, 2518, 2562])
    class_all.append([5, 1, 6, 2702, 2726])
    class_all.append([5, 1, 7, 2770, 2808])
    class_all.append([5, 1, 8, 2952, 3060])
    class_all.append([5, 1, 9, 3096, 3188])

    class_all.append([6, 0, 0, 130, 252])
    class_all.append([6, 0, 1, 1038, 1186])
    class_all.append([6, 0, 2, 1256, 1372])
    class_all.append([6, 0, 3, 1450, 1602])
    class_all.append([6, 0, 4, 1602, 1758])
    class_all.append([6, 0, 5, 2534, 2614])
    class_all.append([6, 0, 6, 3290, 3350])
    class_all.append([6, 0, 7, 3350, 3522])
    class_all.append([6, 0, 8, 3666, 3902])
    class_all.append([6, 0, 9, 3990, 4128])

    class_all.append([6, 1, 0, 552, 638])
    class_all.append([6, 1, 1, 878, 1014])
    class_all.append([6, 1, 2, 1014, 1146])
    class_all.append([6, 1, 3, 1228, 1352])
    class_all.append([6, 1, 4, 1352, 1518])
    class_all.append([6, 1, 5, 1990, 2058])
    class_all.append([6, 1, 6, 2434, 2496])
    class_all.append([6, 1, 7, 2496, 2618])
    class_all.append([6, 1, 8, 2672, 2982])
    class_all.append([6, 1, 9, 3042, 3152])

    class_all.append([7, 0, 0, 446, 534])
    class_all.append([7, 0, 1, 714, 812])
    class_all.append([7, 0, 2, 836, 900])
    class_all.append([7, 0, 3, 1026, 1144])
    class_all.append([7, 0, 4, 1228, 1588])
    class_all.append([7, 0, 5, 1880, 1916])
    class_all.append([7, 0, 6, 2236, 2268])
    class_all.append([7, 0, 7, 2334, 2398])
    class_all.append([7, 0, 8, 2598, 2772])
    class_all.append([7, 0, 9, 2794, 2892])

    class_all.append([7, 1, 0, 138, 246])
    class_all.append([7, 1, 1, 610, 716])
    class_all.append([7, 1, 2, 770, 878])
    class_all.append([7, 1, 3, 1126, 1200])
    class_all.append([7, 1, 4, 1364, 1650])
    class_all.append([7, 1, 5, 1826, 1878])
    class_all.append([7, 1, 6, 2030, 2078])
    class_all.append([7, 1, 7, 2126, 2204])
    class_all.append([7, 1, 8, 2280, 2506])
    class_all.append([7, 1, 9, 2574, 2650])

    class_all.append([8, 0, 0, 404, 544])
    class_all.append([8, 0, 1, 1080, 1196])
    class_all.append([8, 0, 2, 1212, 1290])
    class_all.append([8, 0, 3, 1422, 1538])
    class_all.append([8, 0, 4, 1668, 1970])
    class_all.append([8, 0, 5, 2688, 2728])
    class_all.append([8, 0, 6, 3266, 3316])
    class_all.append([8, 0, 7, 3316, 3390])
    class_all.append([8, 0, 8, 3576, 3762])
    class_all.append([8, 0, 9, 3992, 4118])

    class_all.append([8, 1, 0, 482, 610])
    class_all.append([8, 1, 1, 1026, 1158])
    class_all.append([8, 1, 2, 1206, 1310])
    class_all.append([8, 1, 3, 1546, 1678])
    class_all.append([8, 1, 4, 1714, 2120])
    class_all.append([8, 1, 5, 2466, 2522])
    class_all.append([8, 1, 6, 2696, 2760])
    class_all.append([8, 1, 7, 2770, 2838])
    class_all.append([8, 1, 8, 4708, 4872])
    class_all.append([8, 1, 9, 4904, 4964])

    class_all.append([9, 0, 0, 100, 272])
    class_all.append([9, 0, 1, 562, 730])
    class_all.append([9, 0, 2, 730, 862])
    class_all.append([9, 0, 3, 924, 1150])
    class_all.append([9, 0, 4, 1394, 1846])
    class_all.append([9, 0, 5, 3304, 3388])
    class_all.append([9, 0, 6, 3468, 3524])
    class_all.append([9, 0, 7, 3524, 3608])
    class_all.append([9, 0, 8, 3962, 4222])
    class_all.append([9, 0, 9, 4268, 4336])

    class_all.append([9, 1, 0, 96, 220])
    class_all.append([9, 1, 1, 500, 658])
    class_all.append([9, 1, 2, 664, 770])
    class_all.append([9, 1, 3, 1022, 1232])
    class_all.append([9, 1, 4, 2602, 2784])
    class_all.append([9, 1, 5, 1720, 1810])
    class_all.append([9, 1, 6, 1944, 1994])
    class_all.append([9, 1, 7, 1982, 2062])
    class_all.append([9, 1, 8, 2094, 2350])
    class_all.append([9, 1, 9, 2454, 2598])

    class_all_n = []
    for nseq, info in enumerate(class_all):
        na = info[2]
        info[2] = info[1]
        info[1] = na
        class_all_n.append([nseq] + info)

    return class_all_n
