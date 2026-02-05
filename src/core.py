from __future__ import annotations

import time


class Solution:
    def __init__(self):
        """[summary]."""

    def problem(self, nums: list[int]) -> int:
        tot = 0
        max_tot = 0
        for ea in range(len(nums)):
            tot = max(tot+nums[ea], 0)
            max_tot = max(max_tot, tot)
        return max_tot


if __name__ == "__main__":
    """[summary]"""
    start_time = time.time()
    print(Solution().problem())
    print("--- %s seconds ---" % (time.time() - start_time))
