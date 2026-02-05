from __future__ import annotations

import time


class Solution:
    def __init__(self):
        """[summary]."""

    def problem(self, matrix: list[list[int]]) -> list[int]:
        rows = len(matrix)
        cols = len(matrix[0])
        ans = matrix[0]

        for i in range(1, rows):
            for j in range(cols):
                if j == 0:
                    ans[j] = matrix[i][j] + ans[j]
                elif j == cols - 1:
                    ans[j] = matrix[i][j] + ans[j - 1]
                else:
                    ans[j] = matrix[i][j] + min(ans[j - 1], ans[j])

        return ans




if __name__ == "__main__":
    """[summary]"""
    start_time = time.time()
    print(Solution().problem())
    print("--- %s seconds ---" % (time.time() - start_time))
