from __future__ import annotations

import time


class Solution:
    def __init__(self):
        """[summary]."""

    def problem(self):
        return None

    def multiply(self, num1: str, num2: str) -> str:
        lookup_val = {"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "0":0}
        new_val = 0 # summation of the new value
        num1_len = len(num1) # readability
        num2_len = len(num2) # readability

        if "0" in [num1, num2]:
            return "0" # if one of the numbers is 0, return 0

        # For each digit in each number, convert it to an int by looking it up in the dict.
        # Then multiply these values and add them to the new value.
        # Use the i and j variables to keep track of the position of the digits.
        for i in range(num1_len):
            for j in range(num2_len):
                num1_ch = lookup_val[num1[num1_len-i-1]] # Get the value of the digit with dict lookups.
                num2_ch = lookup_val[num2[num2_len-j-1]]
                """ Alternate way to get the value of the digit:
                num1_ch = ord(num1[num1_len-i-1]) - 48
                num2_ch = ord(num2[num2_len-j-1]) - 48
                """
                new_val+= num1_ch*num2_ch*10**(i+j)

        return str(new_val) # return the new value

if __name__ == "__main__":
    """[summary]"""
    start_time = time.time()
    print(Solution().problem())
    print("--- %s seconds ---" % (time.time() - start_time))
