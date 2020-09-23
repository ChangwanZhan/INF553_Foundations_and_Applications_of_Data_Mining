import sys
from blackbox import BlackBox
# from assignment5.blackbox import BlackBox
import binascii


def myhashs(s):
    hash_values = []
    hash_a = [2433, 18561, 35676, 10151, 66121, 61706, 715, 56495, 24560, 51311, 23282, 37662, 17050, 35323, 11644, 42430]
    hash_b = [16610, 16867, 50851, 30405, 61254, 59559, 39080, 36651, 31055, 22321, 42162, 53335, 14327, 60604, 23421, 51647]
    s = int(binascii.hexlify(s.encode('utf8')), 16)
    for a, b in zip(hash_a, hash_b):
        hash_values.append(((a * s) + b) % 69997)
    return hash_values


def main():
    input_file, output_file = sys.argv[1], sys.argv[4]
    stream_size, num_of_asks = int(sys.argv[2]), int(sys.argv[3])

    hash_function_num = 16
    m = 69997

    bx = BlackBox()
    boom_filter = [0 for _ in range(m)]
    with open(output_file, "w") as f:
        f.write("Time,FPR")
        for time in range(num_of_asks):
            stream_users = bx.ask(input_file, stream_size)
            visited_users = set()
            FP, TN = 0, 0
            for user in stream_users:
                hash_values = myhashs(user)
                count = 0
                for hash_value in hash_values:
                    if boom_filter[hash_value] == 1:
                        count += 1

                if user not in visited_users:
                    if count == hash_function_num:
                        FP += 1
                    else:
                        TN += 1
                    visited_users.add(user)

                for hash_value in hash_values:
                    boom_filter[hash_value] = 1
            FPR = float(FP / (FP + TN))
            f.write("\n{},{}".format(time, FPR))


if __name__ == '__main__':
    main()