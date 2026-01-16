from __future__ import annotations

import itertools
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

    def addTwoNumbers(self, l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
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

    def lengthOfLongestSubstring(self, s: str) -> int:
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

    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
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

    def longestPalindrome(self, s: str) -> str:
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

    def myAtoi(self, s: str) -> int:  # noqa: C901 # Ignoring complexity (inner functions)
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

    def isPalindrome(self, x: int) -> bool:
        """LeetCode #9: Palindrome Number.

        Given an integer x, return true if x is a palindrome, and false otherwise.
        """
        return str(x) == str(x)[::-1] # use strings, if int you can get a - and causes issues.

    def isMatch(self, s: str, p: str) -> bool:
        """LeetCode #10: Regular Expression Matching.

        Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
        '.' Matches any single character.
        '*' Matches zero or more of the preceding element.
        The matching should cover the entire input string (not partial).
        """
        return bool(re.match("^" + p + "$", s))

    def maxArea(self, height: list[int]) -> int:
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

    def intToRoman(self, num: int) -> str:
        """LeetCode #12: Integer to Roman.

        Given an integer, convert it to a roman numeral.
        """
        roman_numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
        "IV": 4, "IX": 9, "XL": 40,"XC": 90, "CD": 400, "CM": 900}
        roman = ""
        for letter, value in dict(sorted(roman_numerals.items(), key=lambda item: item[1], reverse=True)).items():
            while num >= value:
                num -= value
                roman += letter
        return roman

    def romanToInt(self, s: str) -> int:
        """LeetCode #13: Roman to Integer.

        Given a roman numeral, convert it to an integer.
        """
        roman_numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
        "IV": 4, "IX": 9, "XL": 40, "XC": 90, "CD": 400, "CM": 900}

        li = [roman_numerals[x] for x in s]
        tot = 0
        for i in range(len(li)):
            if i < len(li) - 1 and li[i] < li[i + 1]:
                tot -= li[i]
            else:
                tot += li[i]
        return tot

    def longestCommonPrefix(self, strs: list[str]) -> str:
        """LeetCode #14: Longest Common Prefix.

        Write a function to find the longest common prefix string amongst an array of strings.
        If there is no common prefix, return an empty string "".
        """
        prefix = strs[0]
        for i in range(1, len(strs)):
            while strs[i].find(prefix) != 0:
                prefix = prefix[:-1]
        return prefix

    def threeSum(self, nums: list[int]) -> list[list[int]]:
        """LeetCode #15: 3Sum.

        Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
        """
        solutions = []
        nums.sort() # Get it in order so the pointers work

        for i in range(len(nums)-2):
            if nums[i] > 0 or nums[-1] < 0: # No need to look further (all are - or all are +)
                break
            if i > 0 and nums[i] == nums[i-1]: # You're looking at the same as the last one, move on
                continue

            # Initialize the left & right pointers
            left_point = i+1
            right_point = len(nums)-1

            while left_point < right_point:
                sum_val = nums[i] + nums[left_point] + nums[right_point]

                if sum_val > 0: # Too big, move to smaller numbers
                    right_point -= 1
                elif sum_val < 0: # Too small, move to bigger numbers
                    left_point += 1
                else: # Found a solution!
                    solutions.append([nums[i], nums[left_point], nums[right_point]])

                    # Move both so you don't repeat
                    left_point += 1
                    right_point -= 1

                    while left_point < right_point:
                        if nums[left_point-1] == nums[left_point]: # If same as the last one, keep going
                            left_point += 1
                        elif nums[right_point+1] == nums[right_point]: # if same as the last one, keep going
                            right_point -= 1
                        else: # Not a duplicate
                            break
        return solutions

    def threeSumClosest(self, nums: list[int], target: int) -> int:
        """LeetCode #16: 3Sum Closest.

        Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.
        """
        solution = -10_000_000 # Set to a really low number so the first one wins
        nums.sort() # Get it in order so the pointers work

        if sum(nums[:3]) > target: # You ain't gunna hit it, your list is too big
            return sum(nums[:3])
        if sum(nums[-3:]) < target: # You ain't gunna hit it, your list is too small
            return sum(nums[-3:])
        if len(nums) == 3: # You only got 3, and I gotta add 3, so..  # noqa: PLR2004
            return sum(nums)

        for i in range(len(nums)-2): # Fine, we'll do the logic
            if i > 0 and nums[i] == nums[i-1]: # You're looking at the same as the last one, move on
                continue

            # Initialize the left & right pointers
            left_point = i+1
            right_point = len(nums)-1

            while left_point < right_point:
                sum_val = nums[i] + nums[left_point] + nums[right_point]

                diff = target - sum_val

                if abs(diff) < abs(target - solution):
                    solution = sum_val

                if diff < 0: # Too big, move to smaller numbers
                    right_point -= 1
                elif diff > 0: # Too small, move to bigger numbers
                    left_point += 1
                else: # Found a solution!
                    break
        return solution

    def letterCombinations(self, digits: str) -> list[str]:
        """LeetCode #17: Letter Combinations of a Phone Number.

        Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.
        """
        phone = {"2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"}
        li = [phone.get(number) for number in digits] # Get the letters for each digit
        return ["".join(str(item) for item in combo) for combo in itertools.product(*li)] # Generate cartesian product, then join all the touples to strings

    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:  # noqa: C901, PLR0912
        """LeetCode #18: 4Sum.

        Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:
        0 <= a, b, c, d < n
        a, b, c, and d are distinct.
        nums[a] + nums[b] + nums[c] + nums[d] == target
        """
        solutions = []
        nums.sort() # Get it in order so the pointers work
        if len(nums) < 4:  # noqa: PLR2004
            return solutions

        for i in range(len(nums)-3): # You've got 3 right of it
            if i > 0 and nums[i] == nums[i-1]: # You're looking at the same as the last one, move on
                continue

            for j in range(i + 1, len(nums) - 2): # You've got 2 right of it
                if j > i + 1 and nums[j] == nums [j-1]: # You're looking at the same as the last one, move on
                    continue

                # Initialize the left & right pointers
                left_point = j+1
                right_point = len(nums)-1

                while left_point < right_point:
                    sum_val = nums[i] + nums[j] + nums[left_point] + nums[right_point]

                    if sum_val > target: # Too big, move to smaller numbers
                        right_point -= 1
                    elif sum_val < target: # Too small, move to bigger numbers
                        left_point += 1
                    else: # Found a solution!
                        solutions.append([nums[i], nums[j], nums[left_point], nums[right_point]])

                        # Move both so you don't repeat
                        left_point += 1
                        right_point -= 1

                        while left_point < right_point:
                            if nums[left_point-1] == nums[left_point]: # If same as the last one, keep going
                                left_point += 1
                            elif right_point != len(nums)-1 and nums[right_point+1] == nums[right_point]: # if same as the last one, keep going
                                right_point -= 1
                            else: # Not a duplicate
                                break
        return solutions

    def removeNthFromEnd(self, head: [ListNode] | None, n: int) -> [ListNode] | None:
        """LeetCode #19: Remove Nth Node From End of List.

        Given the head of a linked list, remove the nth node from the end of the list and return its head.
        """
        base = ListNode() # Create a base
        base.next = head # Set the base that the pointers will traverse, starting at the head

        lead_point = base # Start at the base
        follow_point = base # Start at the base

        for _ in range(n): # Create the gap
            lead_point = lead_point.next

        while lead_point.next: #Move until you hit the end
            follow_point = follow_point.next
            lead_point = lead_point.next

        follow_point.next = follow_point.next.next # Since we stepped lead ahead, now follow is at the one to skip. Next.next.

        return base.next

    def isValid(self, s: str) -> bool:
        """LeetCode #20: Valid Parentheses.

        Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
        """
        checker = ""

        valid = {
            "(":")",
            "[":"]",
            "{":"}",
            "]":None,
            "}":None,
            ")":None,
        }

        for ch in s:
            if ch in "({[": # You're building up the amount you need to close
                checker+=ch
            elif len(checker) > 0 and ch == valid.get(checker[-1]): # Check you had an open & have the right close
                checker=checker[:-1] #You matched an open & close, remove the open from checker.
            else:
                return False #Checker was blank OR the character didn't have the right closing
        return checker == "" # If you have un-finished items will return False, otherwise True.

    def mergeTwoLists(self, list1: [ListNode] | None, list2: [ListNode] | None) -> [ListNode] | None:
        """LeetCode #21: Merge Two Sorted Lists.

        You are given the heads of two sorted linked lists list1 and list2.
        Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
        Return the head of the merged linked list.
        """
        base = ListNode()
        current_node = base

        if list1 is None or list2 is None: # If a list is blank, return the one that isn't
            return list1 or list2

        while list1 or list2: #While either list has something, keep going
            digit1 = list1.val if list1 else 101 #101 is outside range
            digit2 = list2.val if list2 else 101 #101 is outside range

            if digit1 <= digit2: # Take from list1 when matched, or it's less
                current_node.next = ListNode(digit1)
                list1 = list1.next if list1 else None
            elif digit2 < digit1: # If list2 is lower, take it
                current_node.next = ListNode(digit2)
                list2 = list2.next if list2 else None
            current_node = current_node.next
        return base.next


if __name__ == "__main__":
    sol = Solution()
