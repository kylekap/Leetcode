from __future__ import annotations

import re

import util
from util import ListNode


class Solution:
    def twosum(self, nums: list[int], target: int) -> list[int]:
        """Leetcode #1: Two Sum.

        Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
        You may assume that each input would have exactly one solution, and you may not use the same element twice.
        You can return the answer in any order.
        """
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in nums:
                j = nums.index(diff)
                if j != i:
                    return i, nums.index(diff)
        return None

    def addTwoNumbers(self, l1: ListNode | None, l2: ListNode | None) -> ListNode | None:  # noqa: N802
        """LeetCode #2: Add Two Numbers.

        You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
        You may assume the two numbers do not contain any leading zero, except the number 0 itself.
        """
        answer_node = ListNode()
        current_node = answer_node
        carry = 0
        while l1 or l2 or carry:
            digit1 = l1.val if l1 else 0
            digit2 = l2.val if l2 else 0

            tot = digit1 + digit2 + carry
            carry = tot // 10
            digit_to_add = tot % 10

            current_node.next = ListNode(digit_to_add)
            current_node = current_node.next

            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return answer_node.next

    def lengthOfLongestSubstring(self, s: str) -> int:  # noqa: N802
        """LeetCode #3: Longest Substring Without Repeating Characters.

        Given a string s, find the length of the longest substring without repeating characters.
        """
        used = {}
        max_len = 0
        start = 0

        for i, c in enumerate(s):
            if c in used and start <= used[c]:
                start = used[c]+1
            else:
                max_len = max(max_len, i - start + 1)
            used[c] = i
        return max_len

    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:  # noqa: N802
        """LeetCode #4: Median of Two Sorted Arrays.

        Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
        The overall run time complexity should be O(log (m+n)).
        """
        new = nums1 + nums2
        new.sort()
        arr_size = len(new)
        if arr_size % 2 == 0:
            return (new[arr_size//2-1] + new[arr_size//2]) / 2
        return new[arr_size//2]

    def longestPalindrome(self, s: str) -> str:  # noqa: N802
        """LeetCode #5: Longest Palindromic Substring.

        Given a string s, return the longest palindromic substring in s.
        """
        longest_palindrome = ""
        for i in range(len(s)):
            for j in range(i + len(longest_palindrome), len(s)+1):
                if util.palindrome(s[i:j]):
                    longest_palindrome = s[i:j]
        return longest_palindrome

    def convert(self, s: str, numRows: int) -> str:  # noqa: N803
        """LeetCode #6: ZigZag Conversion.

        The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
        P   A   H   N
        A P L S I I G
        Y   I   R
        And then read line by line: "PAHNAPLSIIGYIR"
        """
        def _problem6_vert(str_list, string_val):
            for i, ea in enumerate(string_val):
                str_list[i] += ea
            return str_list

        def _problem6_diag(str_list, string_val):
            for i, ea in enumerate(string_val):
                str_list[len(str_list)-2-i] += ea
            return str_list

        def _problem6_combine(str_list):
            return "".join(str(item) for item in str_list)

        str_list = ["" for _ in range(numRows)]
        str_reducing = s
        while len(str_reducing) > 0:
            str_list = _problem6_vert(str_list, str_reducing[:numRows])
            str_reducing = str_reducing[numRows:]
            if numRows > 2:  # noqa: PLR2004
                str_list = _problem6_diag(str_list, str_reducing[:numRows-2])
                str_reducing = str_reducing[numRows-2:]
        return _problem6_combine(str_list)

    def reverse(self, x: int) -> int:
        """LeetCode #7: Reverse Integer.

        Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
        """
        val = util.reverse_int(x)
        if abs(val) > 2**31:
            return 0
        return val

    def myAtoi(self, s: str) -> int:  # noqa: C901, N802 # Ignoring complexity (inner functions)
        """LeetCode #8: String to Integer (atoi).

        Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer.
        The algorithm for myAtoi(string s) is as follows:
        1. Whitespace: Read in and ignore any leading whitespace.
        2. Signedness: Check if the next character (if not already at the end of the string) is '-' or '+', assuming positive if neither is present.
        3. Conversion: Read in next the characters until the next non-digit character or the end of the input is reached. If no digits were read, then the integer is 0.
        4. Rounding: If the integer is overflowed, return 2^31 - 1 (2147483647) if the integer is positive, or -2^31 (2147483648) if the integer is negative.
        Return: Return the integer as the final result.
        """

        def _build_number(s):
            val = ""
            first_character_trigger = False
            for i, ea in enumerate(s):
                if val == "":
                    if ea == " " and not first_character_trigger:
                        continue
                    if ea == " " and first_character_trigger:
                        break
                    if ea == "0":
                        first_character_trigger = True
                        continue
                    if ea in "+-" and not first_character_trigger:
                        val+=ea
                        continue
                    if ea in num:
                        val+=ea
                    else:
                        print(i, ea)
                        break
                elif ea in num:
                    val+=ea
                else:
                    break
            return val

        def _check_valid(val):
            if val in {"", "+", "-"}:
                return 0
            if val[0] == "+":
                val = val[1:]
            val = int(val)

            if val > 2**31-1:
                return 2**31-1
            if val < -2**31:
                return -2**31
            return val

        num = "0123456789"
        val = _build_number(s)
        return _check_valid(val)

    def isPalindrome(self, x: int) -> bool:  # noqa: N802
        """LeetCode #9: Palindrome Number.

        Given an integer x, return true if x is a palindrome, and false otherwise.
        """
        return str(x) == str(x)[::-1] # use strings, if int you can get a - and causes issues.

    def isMatch(self, s: str, p: str) -> bool:  # noqa: N802
        """LeetCode #10: Regular Expression Matching.

        Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
        '.' Matches any single character.
        '*' Matches zero or more of the preceding element.
        The matching should cover the entire input string (not partial).
        """
        return bool(re.match("^" + p + "$", s))

    def maxArea(self, height: list[int]) -> int:  # noqa: N802
        """LeetCode #11: Container With Most Water.

        You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
        Find two lines that together with the x-axis form a container, such that the container contains the most water.
        Return the maximum amount of water a container can store.
        Notice that you may not slant the container.
        """
        def _find_area(height1, height2, distance):
            return min(height1, height2)*distance

        max_vol = 0
        p1 = 0
        p2 = len(height)-1
        while p1 < p2:
            max_vol = max(max_vol, _find_area(height[p1], height[p2], p2-p1))
            if height[p1] > height[p2]:
                p2-=1
            else:
                p1+=1
        return max_vol


