class Solution:
    def twosum(self, nums: list[int], target: int) -> list[int]:
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in nums:
                j = nums.index(diff)
                if j != i:
                    return i, nums.index(diff)
        return None

if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9
    print(Solution().twoSum(nums, target))
