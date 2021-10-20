from scipy.stats import mode
import numpy as np

def main():
    a = np.array([[6, 8, 3, 0],
              [3, 2, 1, 7],
              [8, 1, 8, 4],
              [5, 3, 0, 5],
              [4, 7, 5, 9]])
    print(mode(a)[0])

if __name__ == '__main__':
    main()