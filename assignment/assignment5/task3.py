import sys
from blackbox import BlackBox
# from assignment5.blackbox import BlackBox
import random


def main():
    input_file, output_file = sys.argv[1], sys.argv[4]
    stream_size, num_of_asks = int(sys.argv[2]), int(sys.argv[3]) #stream size = 100

    bx = BlackBox()
    seq_num = 0
    window_size = 100
    random.seed(553)
    user_list = []
    with open(output_file, "w") as f:
        f.write("seqnum,0_id,20_id,40_id,60_id,80_id")
        for i in range(num_of_asks):
            stream_users = bx.ask(input_file, stream_size)
            if seq_num == 0:
                user_list += stream_users
                seq_num += stream_size
            else:
                for user in stream_users:
                    seq_num += 1
                    prob = random.randint(0, 100000) % seq_num
                    if prob < window_size:
                        pos = random.randint(0, 100000) % window_size
                        user_list[pos] = user
            f.write("\n{},{},{},{},{},{}".format(seq_num,user_list[0],user_list[20],user_list[40],user_list[60],user_list[80]))
            print("{},{},{},{},{},{}".format(seq_num,user_list[0],user_list[20],user_list[40],user_list[60],user_list[80]))


if __name__ == '__main__':
    main()