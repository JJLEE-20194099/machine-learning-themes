import math


def classifyAPoint(points, p, k=3):
    distance = []
    for group in points:
        for feature in points[group]:
            dis = math.sqrt((feature[0] - p[0]) ** 2 + (feature[1] - p[1]) ** 2)

            distance.append((dis, group))

    distance = sorted(distance)[:k]

    count_label_0 = 0
    count_label_1 = 0

    for d in distance:
        if d[1] == 0:
            count_label_0 += 1
        elif d[1] == 1:
            count_label_1 += 1

    return 0 if count_label_0 > count_label_1 else 1

def main():
    points = {0: [(1, 12), (2, 5), (3, 6), (3, 10), (3.5, 8), (2, 11), (2, 9), (1, 7)],
              1: [(5, 3), (3, 2), (1.5, 9), (7, 2), (6, 1), (3.8, 1), (5.6, 4), (4, 2), (2, 5)]}

    p = (5, 3.5)

    k = 3

    print("The value classified to unknown point is: {}" . format(classifyAPoint(points, p, k)));

if __name__ == '__main__':
    main()


