import time


class Solution:
    def __init__(self):
        """[summary]."""

    def problem(self, nums: list[int], target: int) -> list[list[int]]: #fourSum
        solutions = []
        nums.sort() # Get it in order so the pointers work

        for i in range(len(nums)-3):
            #if nums[i] > 0 or nums[-1] < 0: # No need to look further (all are - or all are +)
            #    break
            if i > 0 and nums[i] == nums[i-1]: # You're looking at the same as the last one, move on
                continue

            # Initialize the left & right pointers
            left_point = i+1
            middle_point = i+2
            right_point = len(nums)-1

            while left_point < right_point:
                sum_val = nums[i] + nums[left_point] + nums[middle_point] + nums[right_point]

                if sum_val > 0: # Too big, move to smaller numbers
                    right_point -= 1
                elif sum_val < 0: # Too small, move to bigger numbers
                    if left_point + 1 == middle_point:
                        middle_point += 1
                    else:
                        left_point += 1
                else: # Found a solution!
                    solutions.append([nums[i], nums[left_point], nums[middle_point], nums[right_point]])

                    # Move both so you don't repeat
                    left_point += 1
                    middle_point = left_point+1
                    right_point -= 1

                    while left_point < right_point:
                        if nums[left_point-1] == nums[left_point]: # If same as the last one, keep going
                            left_point += 1
                        if nums[middle_point-1] == nums[middle_point] or middle_point == left_point: # if same as the last one, keep going
                            middle_point += 1
                        if nums[right_point+1] == nums[right_point]: # if same as the last one, keep going
                            right_point -= 1
                        else: # Not a duplicate
                            break
        return solutions


if __name__ == "__main__":
    """[summary]"""
    start_time = time.time()
    print(Solution().problem([1,0,-1,0,-2,2], 0), "Time: ", time.time() - start_time)
