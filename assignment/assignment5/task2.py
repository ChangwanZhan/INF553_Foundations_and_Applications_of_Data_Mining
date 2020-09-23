import sys
from blackbox import BlackBox
# from assignment5.blackbox import BlackBox
import binascii


def myhashs(s):
    hash_values = []
    hash_a = [4096, 5, 2054, 4101, 4105, 2059, 2060, 13, 4109, 4111, 22, 2075, 4123, 2077, 4126, 4125, 27, 4129, 4130, 28, 4132, 2086, 4136, 4137, 4138, 2091, 4140, 47, 2099, 2100, 4155, 2109, 2111, 4160, 69, 71, 2124, 80, 2129, 2130, 84, 85, 4185, 90, 92, 4190, 2143, 2145, 2146, 2147, 4196, 4194, 4198, 4200, 4201, 106, 4203, 108, 109, 2159, 2160, 114, 2165, 120, 122, 4220, 2173, 4224, 2178, 130, 2180, 4233, 139, 2188, 145, 2194, 148, 4244, 4246, 151, 2196, 153, 2201, 2197, 4249, 2207, 4256, 2215, 4264, 4265, 4266, 2222, 4271, 2224, 4275, 2229, 4278, 186, 2235, 188, 4286, 2239, 191, 2241, 193, 4291, 4294, 2246, 202, 2250, 4303, 208, 2257, 209, 2256, 211, 4306, 2263, 2264, 217, 2267, 4315, 221, 222, 4317, 4316, 2273, 2274, 4323, 4324, 2277, 2278, 2275, 2282, 4331, 4332, 237, 235, 4336, 2288, 242, 244, 2297, 250, 2299, 252, 253, 4346, 254, 256, 4353, 4350, 4356, 261, 2311, 2315, 269, 277, 4373, 4375, 2329, 2330, 4379, 2333, 287, 290, 292, 2342, 300, 4398, 304, 2354, 2359, 2363, 317, 4413, 318, 2368, 4417, 4414, 319, 4419, 4421, 2373, 4424, 328, 329, 2380, 333, 2382, 4431, 4430, 337, 338, 4434, 335, 4437, 4438, 345, 2394, 348, 349, 2397, 2399, 4449, 2402, 353, 4452, 4457, 362, 2410, 2414, 2416, 2417, 2421, 2423, 4478, 2431, 383, 2434, 386, 4493, 397, 4504, 2458, 2459, 411, 410, 2463, 4512, 4513, 4514, 2465, 419, 2472, 2477, 4526, 4527, 4528, 433, 2483, 437, 2487, 4536, 440, 4535, 2491, 4541, 4544, 4546, 2498, 2500, 4550, 2503, 456, 4552, 4554, 4555, 461, 2510, 4559, 464, 4562, 467, 4564, 2517, 2518, 471, 2522, 4571, 4573, 2526, 479, 477, 481, 483, 485, 486, 488, 2537, 489, 4584, 4587, 2538, 2546, 498, 500, 2549, 2550, 4601, 2554, 506, 508, 2557, 2559, 4607, 517, 4614, 2566, 2568, 4617, 2570, 523, 521, 2575, 528, 2579, 2580, 2584, 4633, 4632, 539, 4641, 2594, 4643, 4645, 549, 551, 550, 557, 2605, 4654, 2609, 4659, 567, 4664, 569, 570, 2618, 4668, 2621, 575, 2625, 2627, 4676, 4675, 2630, 582, 583, 4683, 2636, 589, 590, 593, 2643, 596, 2648, 4696, 4698, 2652, 605, 606, 607, 609, 4707, 2660, 617, 2665, 4717, 4718, 623, 4719, 2675, 4723, 4728, 4735, 639, 641, 2692, 4742, 4743, 648, 2695, 650, 4746, 2699, 2701, 4750, 4752, 2705, 4757, 4758, 2713, 4762, 667, 666, 2717, 2720, 672, 2722, 2723, 678, 2728, 2730, 4780, 4782, 4785, 2738, 4787, 2741, 693, 695, 4799, 704, 2753, 2752, 4800, 2757, 2758, 4807, 4813, 718, 722, 726, 4823, 727, 4826, 2778, 732, 737, 738, 2786, 740, 742, 746, 2796, 4845, 2800, 2803, 756, 2804, 2806, 2807, 760, 4851, 2811, 4860, 765, 767, 769, 2820, 772, 4872, 2824, 778, 2826, 779, 4877, 2835, 790, 2840, 792, 4889, 4890, 2844, 798, 2846, 2847, 799, 804, 805, 2854, 809, 2858, 4908, 2861, 814, 813, 2864, 4909, 2866, 2867, 2865, 4918, 2871, 2872, 4920, 826, 2879, 832, 4929, 2886, 839, 838, 2889, 4944, 2896, 4946, 2899, 4948, 849, 4950, 854, 4952, 4961, 4962, 866, 865, 4965, 2918, 4963, 4971, 2925, 2927, 4976, 2928, 881, 2930, 1397, 4983, 4984, 4985, 890, 891, 2942, 4990, 2945, 898, 904, 5004, 910, 2961, 2967, 2969, 5018, 5022, 930, 2978, 932, 5029, 2982, 2984, 2985, 938, 5035, 5037, 5041, 5044, 5045, 950, 951, 952, 3002, 3003, 5055, 5056, 3010, 5062, 3014, 3017, 3020, 973, 3023, 976, 977, 5074, 3024, 5076, 5085, 993, 996, 3047, 999, 1000, 3050, 3052, 5101, 1006, 5102, 5103, 1010, 5110, 5111, 3070, 3071, 3072, 3075, 3076, 3079, 3093, 5143, 1048, 5145, 5146, 3100, 5148, 1054, 3109, 1062, 3111, 5167, 1072, 3121, 3123, 5172, 1077, 1079, 5178, 5181, 5184, 5185, 3139, 3140, 5188, 3145, 5195, 1105, 1109, 5206, 1115, 1116, 1117, 3165, 5218, 3171, 1124, 1126, 5225, 1131, 5228, 3179, 1142, 1151, 3204, 3213, 3214, 5266, 1170, 3220, 5268, 5274, 3228, 5278, 5283, 1189, 5285, 5288, 5289, 3243, 1196, 1198, 3249, 5298, 5299, 3252, 3251, 5302, 5300, 3256, 3259, 1214, 1215, 3266, 5316, 5319, 3271, 1226, 5331, 3283, 1237, 3287, 5336, 3290, 1244, 3293, 5342, 5340, 3292, 3295, 5341, 3297, 5350, 1258, 3306, 3308, 1261, 3310, 5358, 1264, 5361, 3314, 1268, 3316, 3319, 5367, 3321, 3322, 1274, 3328, 3332, 1284, 5385, 5386, 5387, 1293, 5391, 1296, 3343, 5398, 5399, 1303, 1306, 3354, 3358, 3367, 5425, 5429, 3382, 1339, 5435, 3390, 5440, 3393, 1344, 3397, 5450, 3403, 1357, 5455, 3408, 3407, 3412, 1364, 1367, 5465, 3417, 1371, 3420, 3422, 5471, 3423, 3426, 5478, 1382, 1387, 1388, 1389, 1393, 3442, 3443, 1396, 5492, 3446, 3447, 1400, 5496, 3450, 5499, 3452, 1403, 3453, 5497, 3456, 1407, 1410, 5507, 3458, 3464, 3466, 5517, 3470, 5518, 1424, 1425, 1426, 5527, 5532, 1438, 1439, 5535, 5538, 5539, 3496, 5549, 1457, 5553, 5557, 5567, 1474, 1476, 3525, 5574, 5578, 3533, 5581, 3536, 3538, 1490, 3541, 1494, 5589, 1498, 3550, 1505, 1507, 5605, 1510, 1513, 3563, 5611, 3567, 1520, 5620, 3572, 3576, 1533, 5630, 3583, 3586, 5635, 5636, 5640, 5641, 3594, 1548, 1553, 5650, 3602, 3604, 1556, 1558, 1565, 3620, 5670, 5672, 3625, 5677, 3631, 3632, 5680, 1586, 5685, 5687, 5688, 3644, 5692, 1601, 3650, 5701, 3660, 1612, 3667, 3669, 3672, 3675, 3679, 1633, 3685, 5734, 3688, 1640, 1642, 5739, 3690, 1644, 3694, 1647, 3703, 5752, 5766, 1673, 3722, 5771, 3725, 3730, 3732, 5781, 3735, 3737, 5789, 1694, 1699, 5795, 5797, 3747, 1703, 1707, 5804, 1712, 1714, 3762, 3763, 5814, 1718, 5818, 1734, 3783, 5836, 1741, 3791, 5840, 3794, 5847, 3804, 3805, 1764, 1768, 3817, 5866, 5867, 5868, 5879, 1785, 1789, 3839, 5888, 5889, 1795, 1798, 3847, 5897, 1802, 5902, 3854, 5904, 5905, 3857, 5908, 3861, 5910, 5911, 3864, 5914, 3867, 5919, 1824, 5922, 3877, 5928, 3884, 5932, 3887, 5936, 1843, 1845, 1846, 3896, 5945, 1850, 5951, 3903, 5956, 1866, 1867, 1869, 1871, 5969, 3922, 3926, 3927, 1880, 3928, 3929, 1883, 5979, 5982, 5983, 1886, 3937, 3938, 1890, 3940, 1893, 3942, 1899, 1900, 1908, 1910, 1911, 3959, 1921, 1928, 3982, 3984, 3986, 3991, 3995, 3998, 1950, 1961, 4010, 1966, 4023, 1978, 4041, 4043, 4046, 2000, 2003, 2005, 2006, 2015, 2018, 4068, 4070, 4072, 4077, 4078, 4082, 4084]
    hash_b = [2048, 4100, 7, 8, 4103, 2058, 2062, 2063, 18, 2068, 2069, 27, 4129, 2087, 4135, 39, 2088, 42, 2092, 4141, 2096, 4145, 2099, 2102, 4150, 2108, 4157, 4158, 4162, 4163, 4168, 75, 79, 4176, 4179, 4180, 86, 4183, 87, 2141, 97, 98, 101, 2151, 4199, 4207, 4208, 112, 119, 2167, 122, 124, 4221, 125, 2172, 4223, 4226, 2180, 133, 2181, 2183, 2184, 139, 2188, 2193, 4242, 2195, 4243, 2199, 2205, 160, 161, 2208, 164, 4262, 2215, 168, 4264, 4266, 170, 172, 4270, 4272, 2226, 4275, 4276, 2229, 2230, 2233, 4288, 2240, 195, 4291, 2481, 2252, 2253, 4303, 213, 215, 2267, 2268, 4317, 2270, 223, 2274, 2275, 2276, 4325, 2278, 2283, 2285, 2289, 4337, 2292, 2293, 2294, 4342, 4345, 2298, 4348, 254, 2302, 2305, 259, 4356, 260, 263, 266, 2315, 4363, 2319, 4367, 2320, 2323, 276, 279, 281, 4377, 283, 4379, 4381, 286, 4383, 2338, 2340, 293, 4388, 292, 2345, 2348, 4397, 4398, 303, 2353, 306, 307, 308, 311, 2362, 4410, 2366, 319, 320, 4418, 2372, 328, 4425, 2379, 2385, 337, 4435, 4439, 4446, 352, 2404, 2405, 4453, 4456, 361, 2416, 4466, 371, 2421, 381, 4477, 383, 4490, 4493, 2446, 2451, 2452, 4501, 406, 4503, 4500, 409, 2459, 4507, 4510, 415, 416, 417, 418, 4512, 4514, 4517, 2468, 4520, 2474, 2475, 2476, 4526, 4527, 4528, 432, 434, 4531, 4532, 433, 438, 2487, 4536, 441, 436, 4539, 2491, 2489, 446, 2495, 2497, 456, 4554, 2506, 4556, 461, 2511, 2515, 4565, 2518, 4571, 477, 4573, 4577, 2529, 4584, 489, 2538, 4586, 2539, 2542, 4590, 4592, 2547, 4596, 2551, 2555, 509, 4605, 510, 2560, 2566, 4614, 4616, 522, 525, 539, 2588, 4638, 4639, 2590, 4641, 546, 2595, 2596, 4645, 2598, 2599, 551, 2607, 4655, 4657, 562, 563, 564, 561, 566, 570, 2632, 2633, 2634, 586, 591, 2642, 2644, 597, 4694, 598, 4697, 603, 604, 4703, 610, 2663, 2666, 623, 624, 4720, 4722, 627, 631, 2680, 633, 2682, 2683, 2685, 2688, 4739, 2693, 4742, 2696, 2697, 2698, 2701, 2704, 659, 660, 661, 4756, 667, 4764, 669, 2720, 4770, 2723, 4772, 4773, 677, 4775, 679, 681, 2728, 4780, 687, 690, 4787, 693, 4790, 2745, 2746, 4798, 2754, 706, 2756, 709, 4806, 4807, 2759, 2763, 4813, 4814, 2766, 717, 2765, 2773, 725, 730, 731, 732, 2780, 2779, 4831, 736, 2782, 2785, 2786, 740, 2790, 4838, 2793, 747, 4846, 757, 2808, 4857, 762, 763, 2809, 765, 4863, 768, 2818, 771, 2819, 4867, 775, 2825, 778, 4876, 4878, 4879, 784, 4883, 2838, 4888, 792, 794, 4892, 4896, 4897, 4899, 804, 2853, 805, 2859, 811, 812, 4911, 2871, 824, 2879, 2880, 833, 4929, 4934, 4936, 2889, 4938, 4939, 842, 846, 4944, 849, 852, 856, 2905, 2906, 2904, 4954, 863, 4963, 868, 4964, 870, 4968, 4970, 2925, 4975, 880, 4983, 889, 2940, 893, 2948, 903, 5002, 906, 2957, 5006, 5007, 5008, 909, 2958, 914, 918, 2968, 5019, 923, 2974, 927, 2976, 5028, 2981, 5030, 2983, 2988, 5036, 5038, 5039, 2992, 2991, 5042, 949, 2998, 2999, 5049, 5050, 5053, 3006, 3010, 3011, 964, 3016, 969, 5074, 3026, 5082, 3037, 991, 992, 3039, 5096, 1002, 1004, 1008, 1009, 5106, 5105, 1013, 3061, 5115, 5118, 1023, 1024, 3073, 5126, 5130, 3082, 1038, 1039, 3088, 3087, 3090, 1042, 3093, 3096, 1048, 3101, 5152, 1057, 5155, 5156, 5160, 5167, 1072, 1074, 3123, 1078, 5176, 3128, 1080, 5179, 3129, 3134, 1090, 3138, 1095, 3148, 1102, 3151, 1104, 3152, 3159, 1116, 5215, 5216, 3171, 5220, 1125, 3174, 1134, 1138, 5234, 1142, 3191, 1146, 3194, 5244, 5245, 3199, 5251, 3208, 5257, 3209, 3214, 3215, 5262, 3220, 3221, 5270, 3224, 1179, 3227, 1180, 3232, 3236, 5286, 3239, 1194, 5290, 5294, 3247, 1200, 5295, 1202, 3257, 1213, 3263, 5311, 1216, 3269, 1221, 5317, 1229, 1230, 5327, 3280, 5329, 5331, 1237, 5333, 5335, 3287, 3291, 1249, 5349, 5351, 3304, 1256, 3306, 5353, 3315, 1269, 3321, 1274, 5369, 1276, 3325, 3326, 5375, 3329, 5384, 3337, 5387, 5388, 5389, 1293, 3343, 1296, 3344, 5393, 1299, 5396, 5397, 5390, 5394, 5400, 5401, 5402, 1307, 5405, 3358, 3357, 3364, 1316, 3366, 5417, 1322, 1323, 3372, 1328, 5428, 5435, 3388, 3387, 1343, 5440, 3393, 5442, 1347, 5446, 1352, 5453, 3405, 5459, 3413, 5463, 1368, 1369, 5466, 5467, 1378, 1379, 5478, 5482, 1386, 1388, 3437, 5489, 3446, 5495, 5496, 1401, 5497, 3451, 1403, 5499, 5504, 3457, 3462, 3463, 1417, 3468, 1421, 5518, 5521, 3473, 5523, 3476, 1429, 1431, 3488, 3489, 5537, 5540, 3493, 1447, 3496, 1449, 3498, 1451, 5543, 5550, 1455, 3502, 3509, 1463, 3513, 3515, 5564, 5566, 3520, 1473, 1477, 5575, 5580, 5584, 5585, 3539, 5588, 5589, 3545, 5594, 3548, 5599, 1508, 1509, 1510, 5606, 1513, 1514, 5613, 3572, 1530, 5629, 5630, 5632, 3586, 1539, 5637, 1542, 1548, 3598, 5646, 3601, 5652, 3607, 1561, 3611, 3613, 1565, 3618, 3621, 5672, 3626, 3627, 3630, 3634, 3635, 3637, 1590, 1591, 1593, 5691, 3644, 1601, 3651, 1606, 5703, 5704, 1609, 1611, 1612, 5708, 3663, 5716, 1621, 3671, 3672, 3676, 5734, 1639, 1642, 1643, 3692, 1644, 5749, 5750, 1656, 3706, 1660, 5756, 5761, 1671, 1672, 3719, 1675, 5772, 1676, 3726, 1678, 5775, 5779, 1684, 5782, 5784, 3738, 3739, 1690, 3741, 5788, 5792, 1699, 5796, 3750, 5799, 5800, 3755, 1708, 1709, 3759, 1713, 1714, 1716, 5812, 1718, 1720, 1722, 1723, 5820, 3771, 5824, 3780, 1732, 5830, 1736, 5837, 1747, 3797, 3798, 1751, 3799, 5848, 3801, 3803, 5852, 3802, 5857, 3809, 1762, 3812, 5861, 5863, 3816, 5868, 3821, 5870, 1777, 1780, 5877, 1783, 1786, 5883, 1788, 3839, 5888, 5890, 5891, 5892, 1794, 1798, 1799, 5896, 3849, 3851, 5900, 1804, 3854, 3856, 5905, 1809, 1811, 3862, 5910, 3864, 1818, 3870, 1823, 1825, 5922, 1828, 5924, 5926, 5927, 3880, 3884, 1837, 3886, 3887, 1836, 5932, 3892, 1844, 1847, 1848, 1849, 5944, 3899, 5948, 1853, 5950, 3903, 3907, 5956, 1860, 5959, 5968, 3920, 3928, 1881, 3930, 1883, 5976, 5981, 3934, 1886, 1893, 3947, 5996, 1903, 3955, 3956, 1909, 3957, 1914, 1919, 3971, 1929, 1930, 1931, 3991, 1944, 1948, 1952, 1956, 1958, 1960, 1963, 1966, 4017, 1971, 1974, 1975, 4024, 1976, 4026, 1978, 4028, 4027, 4031, 1985, 4036, 4051, 2003, 2006, 4061, 4064, 4065, 2018, 4068, 2029, 4078, 4081, 4082, 2033, 4084, 2035, 2038, 2041, 2042, 4092]
    s = int(binascii.hexlify(s.encode('utf8')), 16)
    for a, b in zip(hash_a, hash_b):
        hash_values.append(((a * s) + b) % 6000)
    return hash_values


def main():
    input_file, output_file = sys.argv[1], sys.argv[4]
    stream_size, num_of_asks = int(sys.argv[2]), int(sys.argv[3])

    hash_function_num = 1000
    groups_len = 10
    hash_functions_per_group = int(hash_function_num / groups_len)

    bx = BlackBox()

    with open(output_file, "w") as f:
        f.write("Time,Ground Truth,Estimation")
        est_all = 0
        gt_all = 0
        for time in range(num_of_asks):
            gt = set()
            stream_users = bx.ask(input_file, stream_size)
            all_hash_values = []
            for user in stream_users:
                gt.add(user)
                hash_values = myhashs(user)
                all_hash_values.append(hash_values)

            estimates = []
            for i in range(hash_function_num):
                longest_trailing_zeros = 0
                for hash_values in all_hash_values:
                    hash_value = hash_values[i]
                    trailing_zeros = 0
                    while hash_value & 1 == 0 and hash_value > 0:
                        trailing_zeros += 1
                        hash_value = hash_value >> 1
                    longest_trailing_zeros = max(trailing_zeros, longest_trailing_zeros)
                estimates.append(2 ** longest_trailing_zeros)

            estimates_avg = []
            for i in range(groups_len):
                sum_est = 0
                for j in range(hash_functions_per_group):
                    sum_est += estimates[i * hash_functions_per_group + j]
                estimates_avg.append(float(sum_est / hash_functions_per_group))
            estimates_avg.sort()
            estimate = round(estimates_avg[int(groups_len / 2)])

            est_all += estimate
            gt_all += len(gt)
            f.write("\n{},{},{}".format(time, len(gt), estimate))


if __name__ == '__main__':
    main()