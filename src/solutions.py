from __future__ import annotations

import fnmatch
import heapq
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
                start = used[c] + 1
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
            return (new[arr_size // 2 - 1] + new[arr_size // 2]) / 2
        return new[arr_size // 2]

    def longestPalindrome(self, s: str) -> str:
        """LeetCode #5: Longest Palindromic Substring.

        Given a string s, return the longest palindromic substring in s.
        """
        longest_palindrome = ""
        for i in range(len(s)):
            for j in range(i + len(longest_palindrome), len(s) + 1):
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
                str_list[len(str_list) - 2 - i] += ea
            return str_list

        def _problem6_combine(str_list):
            return "".join(str(item) for item in str_list)

        str_list = ["" for _ in range(numRows)]
        str_reducing = s
        while len(str_reducing) > 0:
            str_list = _problem6_vert(str_list, str_reducing[:numRows])
            str_reducing = str_reducing[numRows:]
            if numRows > 2:  # noqa: PLR2004
                str_list = _problem6_diag(str_list, str_reducing[: numRows - 2])
                str_reducing = str_reducing[numRows - 2 :]
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
                        val += ea
                        continue
                    if ea in num:
                        val += ea
                    else:
                        print(i, ea)
                        break
                elif ea in num:
                    val += ea
                else:
                    break
            return val

        def _check_valid(val):
            if val in {"", "+", "-"}:
                return 0
            if val[0] == "+":
                val = val[1:]
            val = int(val)

            if val > 2**31 - 1:
                return 2**31 - 1
            if val < -(2**31):
                return -(2**31)
            return val

        num = "0123456789"
        val = _build_number(s)
        return _check_valid(val)

    def isPalindrome(self, x: int) -> bool:
        """LeetCode #9: Palindrome Number.

        Given an integer x, return true if x is a palindrome, and false otherwise.
        """
        return str(x) == str(x)[::-1]  # use strings, if int you can get a - and causes issues.

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
            return min(height1, height2) * distance

        max_vol = 0
        p1 = 0
        p2 = len(height) - 1
        while p1 < p2:
            max_vol = max(max_vol, _find_area(height[p1], height[p2], p2 - p1))
            if height[p1] > height[p2]:
                p2 -= 1
            else:
                p1 += 1
        return max_vol

    def intToRoman(self, num: int) -> str:
        """LeetCode #12: Integer to Roman.

        Given an integer, convert it to a roman numeral.
        """
        roman_numerals = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
            "IV": 4,
            "IX": 9,
            "XL": 40,
            "XC": 90,
            "CD": 400,
            "CM": 900,
        }
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
        roman_numerals = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
            "IV": 4,
            "IX": 9,
            "XL": 40,
            "XC": 90,
            "CD": 400,
            "CM": 900,
        }

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
        nums.sort()  # Get it in order so the pointers work

        for i in range(len(nums) - 2):
            if nums[i] > 0 or nums[-1] < 0:  # No need to look further (all are - or all are +)
                break
            if i > 0 and nums[i] == nums[i - 1]:  # You're looking at the same as the last one, move on
                continue

            # Initialize the left & right pointers
            left_point = i + 1
            right_point = len(nums) - 1

            while left_point < right_point:
                sum_val = nums[i] + nums[left_point] + nums[right_point]

                if sum_val > 0:  # Too big, move to smaller numbers
                    right_point -= 1
                elif sum_val < 0:  # Too small, move to bigger numbers
                    left_point += 1
                else:  # Found a solution!
                    solutions.append([nums[i], nums[left_point], nums[right_point]])

                    # Move both so you don't repeat
                    left_point += 1
                    right_point -= 1

                    while left_point < right_point:
                        if nums[left_point - 1] == nums[left_point]:  # If same as the last one, keep going
                            left_point += 1
                        elif nums[right_point + 1] == nums[right_point]:  # if same as the last one, keep going
                            right_point -= 1
                        else:  # Not a duplicate
                            break
        return solutions

    def threeSumClosest(self, nums: list[int], target: int) -> int:
        """LeetCode #16: 3Sum Closest.

        Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.
        """
        solution = -10_000_000  # Set to a really low number so the first one wins
        nums.sort()  # Get it in order so the pointers work

        if sum(nums[:3]) > target:  # You ain't gunna hit it, your list is too big
            return sum(nums[:3])
        if sum(nums[-3:]) < target:  # You ain't gunna hit it, your list is too small
            return sum(nums[-3:])
        if len(nums) == 3:  # You only got 3, and I gotta add 3, so..  # noqa: PLR2004
            return sum(nums)

        for i in range(len(nums) - 2):  # Fine, we'll do the logic
            if i > 0 and nums[i] == nums[i - 1]:  # You're looking at the same as the last one, move on
                continue

            # Initialize the left & right pointers
            left_point = i + 1
            right_point = len(nums) - 1

            while left_point < right_point:
                sum_val = nums[i] + nums[left_point] + nums[right_point]

                diff = target - sum_val

                if abs(diff) < abs(target - solution):
                    solution = sum_val

                if diff < 0:  # Too big, move to smaller numbers
                    right_point -= 1
                elif diff > 0:  # Too small, move to bigger numbers
                    left_point += 1
                else:  # Found a solution!
                    break
        return solution

    def letterCombinations(self, digits: str) -> list[str]:
        """LeetCode #17: Letter Combinations of a Phone Number.

        Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.
        """
        phone = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        li = [phone.get(number) for number in digits]  # Get the letters for each digit
        return [
            "".join(str(item) for item in combo) for combo in itertools.product(*li)
        ]  # Generate cartesian product, then join all the touples to strings

    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:  # noqa: C901, PLR0912
        """LeetCode #18: 4Sum.

        Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:
        0 <= a, b, c, d < n
        a, b, c, and d are distinct.
        nums[a] + nums[b] + nums[c] + nums[d] == target
        """
        solutions = []
        nums.sort()  # Get it in order so the pointers work
        if len(nums) < 4:  # noqa: PLR2004
            return solutions

        for i in range(len(nums) - 3):  # You've got 3 right of it
            if i > 0 and nums[i] == nums[i - 1]:  # You're looking at the same as the last one, move on
                continue

            for j in range(i + 1, len(nums) - 2):  # You've got 2 right of it
                if j > i + 1 and nums[j] == nums[j - 1]:  # You're looking at the same as the last one, move on
                    continue

                # Initialize the left & right pointers
                left_point = j + 1
                right_point = len(nums) - 1

                while left_point < right_point:
                    sum_val = nums[i] + nums[j] + nums[left_point] + nums[right_point]

                    if sum_val > target:  # Too big, move to smaller numbers
                        right_point -= 1
                    elif sum_val < target:  # Too small, move to bigger numbers
                        left_point += 1
                    else:  # Found a solution!
                        solutions.append([nums[i], nums[j], nums[left_point], nums[right_point]])

                        # Move both so you don't repeat
                        left_point += 1
                        right_point -= 1

                        while left_point < right_point:
                            if nums[left_point - 1] == nums[left_point]:  # If same as the last one, keep going
                                left_point += 1
                            elif (
                                right_point != len(nums) - 1 and nums[right_point + 1] == nums[right_point]
                            ):  # if same as the last one, keep going
                                right_point -= 1
                            else:  # Not a duplicate
                                break
        return solutions

    def removeNthFromEnd(self, head: [ListNode] | None, n: int) -> [ListNode] | None:
        """LeetCode #19: Remove Nth Node From End of List.

        Given the head of a linked list, remove the nth node from the end of the list and return its head.
        """
        base = ListNode()  # Create a base
        base.next = head  # Set the base that the pointers will traverse, starting at the head

        lead_point = base  # Start at the base
        follow_point = base  # Start at the base

        for _ in range(n):  # Create the gap
            lead_point = lead_point.next

        while lead_point.next:  # Move until you hit the end
            follow_point = follow_point.next
            lead_point = lead_point.next

        follow_point.next = (
            follow_point.next.next
        )  # Since we stepped lead ahead, now follow is at the one to skip. Next.next.

        return base.next

    def isValid(self, s: str) -> bool:
        """LeetCode #20: Valid Parentheses.

        Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
        """
        checker = ""

        valid = {
            "(": ")",
            "[": "]",
            "{": "}",
            "]": None,
            "}": None,
            ")": None,
        }

        for ch in s:
            if ch in "({[":  # You're building up the amount you need to close
                checker += ch
            elif len(checker) > 0 and ch == valid.get(checker[-1]):  # Check you had an open & have the right close
                checker = checker[:-1]  # You matched an open & close, remove the open from checker.
            else:
                return False  # Checker was blank OR the character didn't have the right closing
        return checker == ""  # If you have un-finished items will return False, otherwise True.

    def mergeTwoLists(self, list1: [ListNode] | None, list2: [ListNode] | None) -> [ListNode] | None:
        """LeetCode #21: Merge Two Sorted Lists.

        You are given the heads of two sorted linked lists list1 and list2.
        Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
        Return the head of the merged linked list.
        """
        base = ListNode()
        current_node = base

        if list1 is None or list2 is None:  # If a list is blank, return the one that isn't
            return list1 or list2

        while list1 or list2:  # While either list has something, keep going
            digit1 = list1.val if list1 else 101  # 101 is outside range
            digit2 = list2.val if list2 else 101  # 101 is outside range

            if digit1 <= digit2:  # Take from list1 when matched, or it's less
                current_node.next = ListNode(digit1)
                list1 = list1.next if list1 else None
            elif digit2 < digit1:  # If list2 is lower, take it
                current_node.next = ListNode(digit2)
                list2 = list2.next if list2 else None
            current_node = current_node.next
        return base.next

    def generateParenthesis(self, n: int) -> list[str]:
        """LeetCode #22: Generate Parentheses.

        Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
        """

        def backtrack(open_paren, close_paren, curr):
            if open_paren < close_paren or open_paren > n or close_paren > n:  # If it's not valid, move on
                return

            if open_paren == n and close_paren == n:  # It's valid and the length is correct
                answer.append(curr)
                return

            if open_paren < n:  # Try adding a left parenthesis
                backtrack(open_paren + 1, close_paren, curr + "(")
            if close_paren < open_paren:  # Try adding a right parenthesis
                backtrack(open_paren, close_paren + 1, curr + ")")

        answer = []
        backtrack(0, 0, "")  # Start the backtracking, populates answer
        return answer

    def mergeKLists(self, lists: list[[ListNode] | None]) -> [ListNode] | None:
        """LeetCode #23: Merge k Sorted Lists.

        You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
        Merge all the linked-lists into one sorted linked-list and return it.
        """
        pq = []

        for i, li in enumerate(lists):
            head = li
            if head is not None:
                heapq.heappush(pq, (head.val, i, head))

        base = ListNode()
        current_node = base

        while pq:
            _, index, top = heapq.heappop(pq)
            current_node.next = top
            current_node = top

            if top.next is not None:
                heapq.heappush(pq, (top.next.val, index, top.next))

        return base.next

    def swapPairs(self, head: [ListNode] | None) -> [ListNode] | None:
        """LeetCode #24: Swap Nodes in Pairs.

        Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
        """
        base = ListNode()
        current_node = base

        if head is None:
            return head

        while head:  # While list has something, keep going
            digit1 = head.val if head is not None else None  # Get first one to swap
            digit2 = head.next.val if head.next is not None else None  # Get second one to swap

            if digit1 is not None and digit2 is not None:  # If both digits exist, swap
                current_node.next = ListNode(digit2)
                current_node = current_node.next
                current_node.next = ListNode(digit1)
                current_node = current_node.next

            elif digit1 is not None:  # If only 1 exists, keep it
                current_node.next = ListNode(digit1)
                current_node = current_node.next
            elif digit2 is not None:  # If only 2 exists, keep it
                current_node.next = ListNode(digit2)
                current_node = current_node.next

            head = head.next if head else None  # Advance
            head = head.next if head else None  # Advance again

        return base.next

    def reverseKGroup(self, head: [ListNode] | None, k: int) -> [ListNode] | None:
        """LeetCode #25: Reverse Nodes in k-Group.

        Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.
        k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

        You may not alter the values in the list's nodes, only nodes themselves may be changed.
        """
        base = ListNode()
        current_node = base

        if head is None:
            return head

        while head:  # While list has something, keep going
            digit = []
            for _ in range(k):  # Get k number of elements
                h = head.val if head is not None else None
                if h is not None:
                    digit.append(h)  # Add to list
                head = head.next if head else None  # Advance

            if len(digit) != k:  # If we don't have k elements, break!
                for ea in digit:  # Add the ones we have that don't make a full group
                    current_node.next = ListNode(ea)
                    current_node = current_node.next
                break

            for ea in digit[::-1]:  # Add in reverse
                current_node.next = ListNode(ea)
                current_node = current_node.next

        return base.next

    def removeDuplicates(self, nums: list[int]) -> int:
        """LeetCode #26: Remove Duplicates from Sorted Array.

        Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.
        """
        if not nums:  # If the list is empty, there's nothing to do
            return 0

        point = 0  # Pointer to keep track of last placed element

        for val in nums[1:]:  # Go through the list
            if val != nums[point]:  # If it's new, add
                point += 1
                nums[point] = val

        return point + 1  # Return the new length

    def removeElement(self, nums: list[int], val: int) -> int:
        """LeetCode #27: Remove Element.

        Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.
        Then return the number of elements in nums which are not equal to val.
        Consider the number of elements in nums which are not equal to val being k, to get accepted, you need to do the following things:
         - Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
         - Return k.
        """
        if not nums:  # If the list is empty, there's nothing to do
            return 0

        point = 0  # Pointer to keep track of last placed element

        for ea in nums:  # Go through the list
            if ea != val:  # If it's not the bad value, add
                nums[point] = ea  # Add
                point += 1  # Advance

        return point  # Return the new length

    def strStr(self, haystack: str, needle: str) -> int:
        """LeetCode #28: Find the Index of the First Occurrence in a String.

        Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
        """
        if needle in haystack:
            return haystack.index(needle)
        return -1

    def divide(self, dividend: int, divisor: int) -> int:
        """LeetCode #29: Divide Two Integers.

        Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.
        The integer division should truncate toward zero, which means losing its fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would be truncated to -2.
        Return the quotient after dividing dividend by divisor.
        Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [-2**31, 2**31 - 1].
        For this problem, if the quotient is strictly greater than 2**31 - 1, then return 2**31 - 1, and if the quotient is strictly less than -2**31, then return -2**31.
        """

        def limit_check(i):
            min_int = -(2**31)
            max_int = -min_int - 1
            if i < min_int:
                return min_int
            if i > max_int:
                return max_int
            return i

        ans = 0
        sign_pos = (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0)

        # Negative bigger range, so convert
        a = -dividend if dividend > 0 else dividend
        b = -divisor if divisor > 0 else divisor

        if dividend == 0:  # Special case
            return 0
        if abs(divisor) == 1:  # Divide by 1 returns original number, sign adjusted
            return limit_check(abs(dividend)) if sign_pos else limit_check(-abs(dividend))

        while a <= b:
            x = b
            ct = 1
            while x >= (-(2**30)) and a <= (x << 1):  # Shift the divisor until it exceeds dividend, up until overflow
                x <<= 1  # Doubling it
                ct <<= 1  # Double the count
            a -= x
            ans += ct

        return limit_check(ans) if sign_pos else limit_check(-ans)  # Return sign adjusted number

    def findSubstring(self, s: str, words: list[str]) -> list[int]:
        """LeetCode #30: Substring with Concatenation of All Words.

        You are given a string s and an array of strings words. All the strings of words are of the same length.
        A concatenated string is a string that exactly contains all the strings of any permutation of words concatenated.

        Return an array of the starting indices of all the concatenated substrings in s. You may return the answer in any order.
        """

        def split_string_evenly(s, size):  # Split string evenly into a list of substrings of len size
            return [s[i : i + size] for i in range(0, len(s), size)]

        word_ct = len(words)
        word_len = len(words[0])
        # Loop through all possible substrings, checking if they contain all words.
        # To check they contain all words, split the string evenly into substrings of the same length & compare against the given word list
        return [
            i
            for i in range(len(s) - word_ct * word_len + 1)
            if sorted(split_string_evenly(s[i : i + word_ct * word_len], word_len)) == sorted(words)
        ]

    def nextPermutation(self, nums: list[int]) -> None:
        """LeetCode #31: Next Permutation.

        A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

        For example, for arr = [1,2,3], the following are considered permutations: [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
        The next permutation of an array of integers is the next lexicographically greater permutation of its integer.
        More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container.
        If such an arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).
        Given an array of integers nums, find the next permutation of nums.
        The replacement must be in place and use only constant extra memory.
        """
        pivot = -1
        num_len = len(nums)  # Just makes it easier

        if num_len == 1:  # Nothing needed
            pass
        elif nums == sorted(nums)[::-1]:  # If list is same as it's reverse, loop back to sorted version
            nums.sort()
        else:
            for i in range(num_len - 2, -1, -1):  # Find where it's no longer asc going from R->L
                if nums[i] < nums[i + 1]:
                    pivot = i
                    break
            for j in range(num_len - 1, pivot, -1):  # Find where the next biggest value than the pivot is
                if nums[j] > nums[pivot]:
                    nums[pivot], nums[j] = nums[j], nums[pivot]  # Swap
                    break

            nums[pivot + 1 :] = reversed(nums[pivot + 1 :])  # Reverse the back half to be the smallest possible

    def longestValidParentheses(self, s: str) -> int:
        """LeetCode #32: Longest Valid Parentheses.

        Given a string containing just the characters '(' and ')', return the length of the longest valid (well-formed) parentheses substring.
        """

        def add_p(ch, open_p, closed_p):
            return (open_p + 1, closed_p) if ch == "(" else (open_p, closed_p + 1)

        max_len = open_p = closed_p = 0

        if len(s) <= 2:  # Base case  # noqa: PLR2004
            return 2 if s == "()" else 0  # Return 2 if valid, else 0
        if s.count("(") == 0 or s.count(")") == 0:  # If you don't have any open or closed parentheses, invalid
            return 0

        for char in s:  # Count the number of open and closed parentheses
            open_p, closed_p = add_p(char, open_p, closed_p)
            if open_p == closed_p:  # If they're equal, it's valid
                max_len = max(max_len, open_p + closed_p)  # Update the max
            elif open_p < closed_p:  # If open is less than closed, it's invalid
                open_p = closed_p = 0  # Reset

        open_p = closed_p = 0

        for char in s[::-1]:  # Count the number of open and closed parentheses, REVERSED THIS TIME
            open_p, closed_p = add_p(char, open_p, closed_p)
            if open_p == closed_p:  # If they're equal, it's valid
                max_len = max(max_len, open_p + closed_p)  # Update the max
            elif open_p > closed_p:  # If open is more than closed, it's invalid
                open_p = closed_p = 0  # Reset
        return max_len

    def search(self, nums: list[int], target: int) -> int:
        """LeetCode #33: Search in Rotated Sorted Array.

        There is an integer array nums sorted in ascending order (with distinct values).
        Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]
        Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
        You must write an algorithm with O(log n) runtime complexity.
        """
        len_nums = len(nums)
        if len_nums < 1 or (len_nums == 1 and nums[0] != target):
            return -1

        left_point = 0
        right_point = len_nums - 1

        while left_point <= right_point:  # Two pointers
            mid_point = (left_point + right_point) // 2  # Get the middle point
            if nums[0] <= nums[mid_point]:  # Left side is sorted
                if nums[0] <= target <= nums[mid_point]:  # Target is in left side
                    right_point = mid_point  # Move the right pointer to the middle to reduce the search size
                else:
                    left_point = (
                        mid_point + 1
                    )  # Move the left pointer to the middle to reduce the search size if it's on the right side
            # Else, the right side is sorted
            elif nums[mid_point] <= target <= nums[len_nums - 1]:  # Target is in right side somewhere
                left_point = mid_point + 1  # Move the left pointer to the middle to reduce the search size
            else:
                right_point = mid_point  # Move the right pointer to the middle to reduce the search size

        return left_point if nums[left_point] == target else -1  # Return the index if it's the target, else -1


    def searchRange(self, nums: list[int], target: int) -> list[int]:
        """LeetCode #34: Find First and Last Position of Element in Sorted Array.

        Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.
        If target is not found in the array, return [-1, -1].
        You must write an algorithm with O(log n) runtime complexity.
        """
        if len(nums) < 1 or target not in nums:
            return [-1, -1]
        return [util.binary_search(nums, target, "first", -1), util.binary_search(nums, target, "last", -1)]

    def searchInsert(self, nums: list[int], target: int) -> int:
        """LeetCode #35: Search Insert Position.

        Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
        You must write an algorithm with O(log n) runtime complexity.
        """
        if target < nums[0]:  # Target is smaller than anything in list
            return 0
        if target > nums[-1]:  # Target is bigger than anything in list
            return len(nums)

        left_point = 0
        right_point = len(nums) - 1

        while left_point <= right_point:  # Two pointers
            mid = (left_point + right_point) // 2  # Find the middle
            if nums[mid] == target or left_point == right_point:  # Found the target, or we're at the end of the search
                return mid
            if target < nums[mid]:  # Need to move earlier in list
                right_point = mid
            else:  # Need to move later in list
                left_point = mid + 1
        return mid  # Return the last found index

    def isValidSudoku(self, board: list[list[str]]) -> bool:
        """LeetCode #36: Valid Sudoku.

        Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

        - Each row must contain the digits 1-9 without repetition.
        - Each column must contain the digits 1-9 without repetition.
        - Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
        Note-

        A Sudoku board (partially filled) could be valid but is not necessarily solvable.
        Only the filled cells need to be validated according to the mentioned rules.
        """

        def check_box(board, cell_row, cell_col):
            """Check a 3x3 box for duplicates."""
            li = [
                board[3 * (cell_row // 3) + i][3 * (cell_col // 3) + j]
                for i, j in itertools.product(range(3), range(3))
            ]  # Get the 3x3 box coordinates
            return util.has_duplicates([x for x in li if x != "."])  # Check for duplicates

        corners = [0, 3, 6]  # The corners of each 3x3 box in the board

        if any(
            util.has_duplicates([x for x in row if x != "."]) for row in board
        ):  # Loop through rows & check none have duplicates
            return False
        if any(
            util.has_duplicates([row[i] for row in board if row[i] != "."]) for i in range(len(board[0]))
        ):  # Loop through columns & check none have duplicates
            return False
        return all(
            not check_box(board, row, col) for row, col in itertools.product(corners, corners)
        )  # Loop through boxes & check none have duplicates. If valid, return True

    def solveSudoku(self, board: list[list[str]]) -> None:
        """LeetCode #37: Sudoku Solver.

        Write a program to solve a Sudoku puzzle by filling the empty cells.
        A sudoku solution must satisfy all of the following rules:
        - Each of the digits 1-9 must occur exactly once in each self.row.
        - Each of the digits 1-9 must occur exactly once in each column.
        - Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
        The '.' character indicates empty cells.
        """

        def _sudoku_brute_force(empty_index):
            if empty_index == len(self.empty_cells):  # If there are no more empty cells
                self.ok = True  # The board is valid
                return
            i, j = self.empty_cells[empty_index]  # Get the coordinates of the empty cell
            b = (i // 3) * 3 + j // 3  # Get the block index
            for v in (
                set("123456789") - self.row[i] - self.col[j] - self.block[b]
            ):  # Check if the value is not in the row, column or 3x3 block
                self.row[i].add(v)
                self.col[j].add(v)
                self.block[b].add(v)
                board[i][j] = v  # Place the value in the cell
                _sudoku_brute_force(empty_index + 1)  # Recursively call the function
                if self.ok:  # If the solution is valid, the flag will be set to True & exit the function.
                    return
                # If the solution is not valid, reset
                board[i][j] = "."
                self.row[i].remove(v)
                self.col[j].remove(v)
                self.block[b].remove(v)
            return

        self.row = [set() for _ in range(9)]  # Build 9 row objects
        self.col = [set() for _ in range(9)]  # Build 9 column objects
        self.block = [set() for _ in range(9)]  # Build 9 3x3 block objects
        self.ok = False  # Flag to check if the board is valid
        self.empty_cells = []  # List of empty cells

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    self.empty_cells.append((i, j))
                else:  # If the cell is not empty
                    v = board[i][j]
                    self.row[i].add(v)
                    self.col[j].add(v)
                    self.block[(i // 3) * 3 + j // 3].add(v)
        _sudoku_brute_force(0)
        return board

    def countAndSay(self, n: int) -> str:
        """LeetCode #38: Count and Say.

        The count-and-say sequence is a sequence of digit strings defined by the recursive formula:
        - countAndSay(1) = "1"
        - countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.

        Run Length Encoding (RLE) is a string compression algorithm that works by replacing consecutive identical characters (repeated 2 or more times) with the concatenation of the character and the number marking the count of the characters (length of the run).
        """

        def _rle_generate(iterable):
            """Generate the RLE string."""
            li = []
            val = ""
            ct = 0
            for i in iterable:  # Iterate through the string
                if i == val:  # If the value is the same
                    ct += 1  # Add 1
                else:
                    li.append((ct, val))  # Append the tuple of the just completed count and value
                    val = i  # Update the value
                    ct = 1  # Reset the count
            li.append((ct, val))  # Append the tuple of the last count and value
            return "".join(
                f"{count}{char}" for count, char in li[1:]
            )  # Return the combined string, ignoring the first blank touple

        answer = "1"

        if n == 1:  # Edge case
            return answer
        for _ in range(2, n + 1):  # Iterate for the 2nd to the nth time
            answer = _rle_generate(answer)
        return answer

    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        """LeetCode #39: Combination Sum.

        Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.
        The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.
        It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
        """
        def backtrack(total, running_list, ix):
            if total == self.target:  # If you've found a total, add the list & stop
                self.answer_li.append(running_list)
                return
            if total > self.target:  # Too big, stop
                return
            for i in range(ix, len(self.candidates)):  # Go through rest of candidates
                backtrack(total + self.candidates[i], [*running_list, self.candidates[i]], i)

        self.answer_li = [[target]] if target in candidates else []  # Set up + grab the "target in list" condition
        self.target = target  # Accessible anywhere
        self.candidates = [x for x in candidates if x < target]  # Only need to deal with the < target values
        backtrack(0, [], 0)  # Backtracking solution
        return self.answer_li

    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        """LeetCode #40: Combination Sum II.

        Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
        Each number in candidates may only be used once in the combination.
        Note: The solution set must not contain duplicate combinations.
        """

        def backtrack(start_index, total):
            if total == target:  # If you've found a total, add the list & stop
                self.answer_li.append(running_list[:])
                return
            for i in range(start_index, len(self.candidates)):  # Go through rest of candidates
                if i != start_index and self.candidates[i] == self.candidates[i - 1]:  # Avoid duplicates
                    continue
                if total + self.candidates[i] > target:  # Check you won't get too big
                    return
                running_list.append(self.candidates[i])
                backtrack(i + 1, total + self.candidates[i])
                running_list.pop()
            return

        self.answer_li = [[target]] if target in candidates else []  # Set up + grab the "target in list" condition
        self.candidates = [
            x for x in candidates if x < target
        ]  # Only need to deal with the candidates who can actually contribute
        self.candidates.sort()  # Sort the candidates

        running_list = []
        backtrack(0, 0)  # Backtracking solution
        return self.answer_li

    def firstMissingPositive(self, nums: list[int]) -> int:
        """LeetCode #41: First Missing Positive.

        Given an unsorted integer array nums, return the smallest missing positive integer.

        You must implement an algorithm that runs in O(n) time and uses constant extra space.
        """
        # Use a set to keep track of the numbers for fastest lookup.
        # Go up to the length of the array, and if the number isn't in the set, return it. Otherwise return the length+1
        s_nums = set(nums)
        for i in range(1, len(nums) + 1):
            if i not in s_nums:
                return i
        return i + 1

    def trap(self, height: list[int]) -> int:
        """LeetCode #42: Trapping Rain Water.

        Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
        """
        # So, we're goin to use two arrays, one forward and one backward. The forward array will have the max value as we walk forward, and the backward array will have the max value as we walk backward.
        # Then, we can compare the forward and backward arrays, and add the min of the two to the total.
        # The "highest" wall from the left will be the max of forward, and the "highest" wall from the right will be the max of backward, then we can add the min of the two since you can't go above the lower wall.

        h_len = len(height)  # How long is array
        forward = [0] * h_len  # Forward array
        backward = [0] * h_len  # Backward array
        max_val_f = 0  # Max value as we walk forward
        max_val_r = 0  # Max value as we walk backward
        total = 0  # Total "water" held

        for i in range(h_len):  # Loop through array
            max_val_f = max(max_val_f, height[i])  # Get max value, forward
            forward[i] = max_val_f  # Add max value to forward array
            max_val_r = max(max_val_r, height[h_len - i - 1])  # Get max value, backward.
            backward[i] = max_val_r  # Add max value to backward array

        backward = backward[::-1]  # Reverse, for easy comparison of i values

        for i in range(len(height)):
            total += max(
                0, min(forward[i], backward[i]) - height[i]
            )  # Add to total. Min of forward and backward - current height. 0 if less than 0
        return total

    def multiply(self, num1: str, num2: str) -> str:
        """LeetCode #43: Multiply Strings.

        Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
        Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
        """
        # Use a dict to lookup the value of each digit.
        # Use two for loops, one for each number, to get the value of each digit and multiply them.
        # Sum it up, and return as a string

        lookup_val = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "0": 0}
        new_val = 0  # summation of the new value
        num1_len = len(num1)  # readability
        num2_len = len(num2)  # readability

        if "0" in [num1, num2]:
            return "0"  # if one of the numbers is 0, return 0

        # For each digit in each number, convert it to an int by looking it up in the dict.
        # Then multiply these values and add them to the new value.
        # Use the i and j variables to keep track of the position of the digits.
        for i in range(num1_len):
            for j in range(num2_len):
                num1_ch = lookup_val[num1[num1_len - i - 1]]  # Get the value of the digit with dict lookups.
                num2_ch = lookup_val[num2[num2_len - j - 1]]
                """ Alternate way to get the value of the digit:
                num1_ch = ord(num1[num1_len-i-1]) - 48
                num2_ch = ord(num2[num2_len-j-1]) - 48
                """
                new_val += num1_ch * num2_ch * 10 ** (i + j)
        """return str(int(num1)*int(num2)) <-- This is the "ignore the rules" way of doing it everyone seems to do."""
        return str(new_val)  # return the new value


    """def isMatch(self, s: str, p: str) -> bool: #TODO(#2): Try without fnmatch
        # LeetCode #44: Wildcard Matching
        return False"""


    def jump(self, nums: list[int]) -> int:
        """LeetCode #45: Jump Game II.

        You are given a 0-indexed array of integers nums of length n. You are initially positioned at index 0.
        Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at index i, you can jump to any index (i+j) where 0 <= j <= nums[i] and i+j < n.
        Return the minimum number of jumps to reach nums[n - 1].
        """
        # Alright, admittedly not the most optimized, but it works.
        # The idea is to use a set to keep track of the indexes we've already visited, and then use a while loop to keep generating indexes until we reach the last index.
        if len(nums) <= 1:  # Edge case
            return 0
        ix = {0}  # Set to keep track of indexes
        count = 0  # Loop #
        while len(nums) - 1 not in ix:
            count += 1  # Add to the loop count
            ix = {
                i + j for i in ix for j in range(1, nums[i] + 1)
            }  # Generate new indexes we can reach with this count of jumps
        return count

    def permute(self, nums: list[int]) -> list[list[int]]: #TODO(#3): Try without itertools
        """LeetCode #46: Permutations.

        Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
        """
        return list(itertools.permutations(nums))

    def permuteUnique(self, nums: list[int]) -> list[list[int]]: #TODO(#4): Try without itertools
        """LeetCode #47: Permutations II.

        Given an array of distinct integers nums, return all the possible permutations. You can return the answer in any order.
        """
        return list(set(itertools.permutations(nums)))

    def rotate(self, matrix: list[list[int]]) -> None:
        """Leetcode #48: Rotate Image.

        You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
        You have to rotate the image in-place, which means you have to modify the input 2D matrix directly.
        DO NOT return anything, modify matrix in-place instead.
        """
        # Swap row & column, iterating backwards for columns to serve as the reversal.
        # Need to do matrix[:] to make the swap inplace
        matrix[:] = [[matrix[j][i] for j in range(len(matrix) - 1, -1, -1)] for i in range(len(matrix[0]))]

        """Alternate way:
        matrix.reverse() # Reverse the matrix
        for row in range(len(A)): # Iterate through rows & swap the columns
            for col in range(row):
                A[row][col], A[col][row] = A[col][row], A[row][col]
        """

    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        """LeetCode #49: Group Anagrams.

        Given an array of strings strs, group the anagrams together. You can return the answer in any order.
        """
        # Convert everything to it's sorted version. Anagrams will have the same sorted version.
        # Generate a dict with the sorted string as the key, and the list of anagrams as the value
        di = {}
        for ea in strs:  # Iterate through each string
            sorted_string = "".join(sorted(ea))  # Get the sorted string (anagrams will have the same sorted string)
            di[sorted_string] = [*di.get(sorted_string, []), ea]  # Add to the dict, using the sorted string as the key
        return list(di.values())  # Return the values, as a list of lists.

    def myPow(self, x: float, n: int) -> float: #TODO(#6): Solve the intended way
        """LeetCode #50: Pow(x, n).

        Implement pow(x, n), which calculates x raised to the power n.
        """
        return 1 if x == 1 or n == 0 else x**n

    def solveNQueens(self, n: int) -> list[list[str]]:
        """LeetCode #51: N-Queens.

        The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.
        Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.
        """
        # Similar to Sudoku -> Use a list of the row, column and diagonal values to check for conflicts.
        # Then use a backtracking algorithm to generate all possible solutions.
        solution = []
        row_tf = [False] * n
        col_tf = [False] * n
        diag1_tf = [False] * (2 * n)
        diag2_tf = [False] * (2 * n)
        board = [
            ["." for i in range(n)] for _ in range(n)
        ]  # Empty board. We're not actually tracking the queens here, it's just for display

        def _brute_force(row):
            if row == n:
                solution.append(["".join(row_list) for row_list in board])  # Add the board to the solution
                return
            for col in range(n):  # Iterate through each column to place the queen
                if (
                    row_tf[row] or col_tf[col] or diag1_tf[row - col] or diag2_tf[row + col]
                ):  # Check if any row/col/diag already has a queen
                    continue  # skip if so
                board[row][col] = "Q"  # Try placing queen
                # Update the row/col/diags
                row_tf[row] = True
                col_tf[col] = True
                diag1_tf[row - col] = True
                diag2_tf[row + col] = True

                # Recursion
                _brute_force(row + 1)

                # Backtrack
                board[row][col] = "."
                row_tf[row] = False
                col_tf[col] = False
                diag1_tf[row - col] = False
                diag2_tf[row + col] = False

        _brute_force(0)  # Start at row 0
        return solution

    def totalNQueens(self, n: int) -> int:
        """LeetCode #52: N-Queens II.

        Given an integer n, return the number of distinct solutions to the n-queens puzzle.
        """
        # Just count the number of solutions to the previous problem
        return len(self.solveNQueens(n))

    def maxSubArray(self, nums: list[int]) -> int:
        """LeetCode #53: Maximum Subarray.

        Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
        """
        # Iterate through - if the running total ever hits negative (aka, won't make anything else added bigger), reset. Return the maximum running total found.
        if max(nums) < 0:  # If all negative, return the max
            return max(nums)
        tot = 0
        max_tot = -1e5
        for ea in nums:  # Iterate through each element
            tot = max(tot + ea, 0)  # Update the running total
            max_tot = max(max_tot, tot)  # Update the max
        return max_tot  # Return the max

    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        """LeetCode #54: Spiral Matrix.

        Given an m x n matrix, return all elements of the matrix in spiral order.
        """

        # Can use the logic from the rotate above -> Move the matrix counterclockwise & keep taking the top row off.abs
        def rotate_clockwise(matrix):
            """Flips a matrix 90 degrees clockwise."""
            return list(zip(*matrix[::-1]))

        def rotate_counterclockwise(matrix):
            """Flips a matrix 90 degrees counterclockwise."""
            return list(zip(*matrix))[::-1]

        ans = []
        while len(matrix) > 0:  # While there's original matrix left
            ans += matrix[0]  # Add the top row
            matrix = rotate_counterclockwise(
                matrix[1:]
            )  # Remove top row & rotate 90degrees, to get the next side to add
        return ans  # Once done, return the list.

    def canJump(self, nums: list[int]) -> bool: # TODO(#7): Implement
        """LeetCode #55: Jump Game.

        You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
        Return true if you can reach the last index, or false otherwise.
        """
        return None


    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        """LeetCode #56: Merge Intervals.

        Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
        """

        # Sort the intervals by start, then check for overlaps (if the end of the previous interval is >= the start of the next interval).
        def overlaps(x, y):
            return max(x[0], y[0]) <= min(x[1], y[1])

        intervals = sorted(intervals)  # Sort

        ans = [intervals[0]]  # Add the first

        for ea in intervals[1:]:  # Iterate each interval
            if overlaps(ea, ans[-1]):  # If overlap
                ans[-1][1] = max(ans[-1][1], ea[1])  # Update the end of the last interval with the new end
            else:
                ans.append(ea)  # Add the new interval if no overlap
        return ans

    def insert(self, intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:  # noqa: N803
        """LeetCode #57: Insert Interval.

        Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
        """
        # Just slap the new interval in there & re-run merge.
        return self.merge([*intervals, newInterval])

    def lengthOfLastWord(self, s: str) -> int:
        """LeetCode #58: Length of Last Word.

        Given a string s consisting of some words separated by spaces, return the length of the last word in the string.
        A word is a maximal substring consisting of non-space characters only.
        """
        rev_s = s[::-1].lstrip()  # Reverse & remove leading whitespace
        words = rev_s.split(" ")  # Split into words
        return len(words[0])  # Return the length of the last word

    def generateMatrix(self, n: int) -> list[list[int]]:
        """LeetCode #59: Spiral Matrix II.

        Given a positive integer n, generate an n x n matrix filled with elements from 1 to n^2 in spiral order.
        """

        # Same logic as Spiral Matrix, but opposite. Start from a single cell & add to the outside, rotating clockwise each time.
        def rotate_clockwise(matrix):
            """Flips a matrix 90 degrees clockwise."""
            return list(zip(*matrix[::-1]))

        i = n**2
        matrix = [[i]]  # Start with a single cell
        while i > 1:  # While there's more to add
            for row in range(len(matrix)):  # Iterate through each row
                i -= 1  # Decrement
                matrix[row] = [i] + list(matrix[row])  # Add to the left of each row. # noqa: RUF005
            matrix = rotate_clockwise(matrix)  # Rotate clockwise to start next iteration
        return matrix  # Multiples of 4, so end up oriented right.

    def getPermutation(self, n: int, k: int) -> str: #TODO(#8): Implement
        """LeetCode #60: Permutation Sequence.

        The set [1,2,3,...,n] contains a total of n! unique permutations.
        By listing and labeling all of the permutations in order, we get the following sequence for n = 3:
        "123"
        "132"
        "213"
        "231"
        "312"
        "321"
        Given n and k, return the kth permutation sequence.
        """
        return None


    def rotateRight(self, head: [ListNode] | None, k: int) -> [ListNode] | None:
        """LeetCode #61: Rotate List.

        Given the head of a linked list, rotate the list to the right by k places.
        """
        # Get the length of the linked list, find how many actual rotations to do, set up two pointers & loop front to back.
        base = head
        li_len = 0

        while base: # get length, to see if we have to loop or anything crazy
            li_len+=1
            base = base.next

        if not head or li_len == 0 or k%li_len == 0: # Edge cases
            return head

        lead = head
        follow = head

        for _ in range(k % li_len): # Step forward the number of spots you need
            lead = lead.next

        while lead.next: # Walk forward to get the "follow" point to the right spot in the rotation & "lead" pointer at the end of the linked list, for easy attachment to beginning
            lead = lead.next
            follow = follow.next

        ans = follow.next # create ans
        follow.next = None # break connection in the second part
        lead.next = head # connect back to front
        return ans

    def uniquePaths(self, m: int, n: int) -> int:
        """LeetCode #62: Unique Paths.

        There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.
        Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
        """
        # This is just a binomial coefficient question, so use that. Same kind of issue as Project Euler #15, but with the variable X & Y, and robot starting on the grid (-1's)
        return int(util.binomial_coefficient(m-1+n-1,n-1))


    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:  # noqa: N803
        """LeetCode #63: Unique Paths II.

        You are given an m x n integer array grid.
        There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.
        An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.
        Return the number of possible unique paths that the robot can take to reach the bottom-right corner.
        The testcases are generated so that the answer will be less than or equal to 2 * 109.
        """
        # Go through each row & add up & right. If an obstacle, ignore. Special cases taken care of first & then go through first row & column to set up for success
        rows = len(obstacleGrid)
        cols = len(obstacleGrid[0])

        if (obstacleGrid[-1][-1] == 1 or obstacleGrid[0][0] == 1) \
            or (rows == 1 and 1 in obstacleGrid[0]) \
            or (cols == 1 and 1 in obstacleGrid):
            return 0

        for j in range(cols): # First row
            if obstacleGrid[0][j] == 1:
                break # If the current cell is an obstacle, ignore it & everything across as inaccessible
            obstacleGrid[0][j] = -1

        for i in range(rows): # First column
            if obstacleGrid[i][0] == 1:
                break # If the current cell is an obstacle, ignore it & everything down as inaccessible
            obstacleGrid[i][0] = -1

        for i in range(1, rows):
            for j in range(1, cols):
                # If the current cell is an obstacle, ignore it. If BOTH the left & up are obstacles, ignore
                if (obstacleGrid[i][j] == 1) or (obstacleGrid[i-1][j] == 1 and obstacleGrid[i][j-1] == 1):
                    continue
                # If the current cell is not an obstacle, add the left & up. The min here is because we're all negative, anything positive is "blocked".
                obstacleGrid[i][j] = min(obstacleGrid[i-1][j],0) + min(obstacleGrid[i][j-1],0)
        return -1*obstacleGrid[-1][-1]

    def minPathSum(self, grid: list[list[int]]) -> int:
        """LeetCode #64: Minimum Path Sum.

        Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
        Note: You can only move either down or right at any point in time.
        """
        # Very similar to approach in #63, but now we have to find the min of the left & up. There's also no obstacles to worry about
        rows = len(grid) # Number of rows
        cols = len(grid[0]) # Number of columns

        # Since they don't have a second cell to add, special cases top row & column
        for row in range(1, rows): # First row
            grid[row][0] += grid[row-1][0]
        for col in range(1, cols): # First column
            grid[0][col] += grid[0][col-1]


        for i in range(1, rows):
            for j in range(1, cols):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1]) # Find min of up & left, add it to current cells value. Since we're going down & right, we'll have already compiled both those previously
        return grid[-1][-1] # Last cell

    def isNumber(self, s: str) -> bool:
        """LeetCode #65: Valid Number.

        Given a string s, determine if it is a number.
        """
        # This is sloppy, but it works?
        # Test cases could use some work, but this met the requirements as posted.
        def check_dot(): # Check if the number is valid around decimals
            print(self.s)
            if len(self.s) == 1 or util.ch_ct_str(self.s, ".") > 1:
                return False
            return not ("e" in self.s and self.s.find("e") < self.s.find("."))

        def check_e(): # Check if the number is valid around e
            if len(self.s) == 1 or util.ch_ct_str(self.s, "e") > 1:
                return False
            li = self.s.split("e")
            return (self.isNumber(li[0]) and self.isNumber(li[1]))

        def remove_plus_minus(s): # Remove any plus or minus signs at the beginning, or around e
            new_s = ""
            for i in range(len(s)):
                if (s[i] == "+" or s[i] == "-") and (i == 0 or s[i-1] == "e"):
                    continue
                new_s+=s[i]
            return new_s

        self.s = remove_plus_minus(s.lower()) # Remove any plus or minus signs at the beginning, or around e
        if util.check_any(self.s, "abcdfghijklmnopqrstuvwxyz+-") or self.s == "": # If the string contains any letters, any more +/-, or is empty, invalid.
            return False
        if "." in self.s and not check_dot(): # If the number has a decimal and the number is not valid around decimals, invalid
            return False
        return not ("e" in self.s and not check_e()) # If the number has an e and the number is not valid around e, invalid

    def plusOne(self, digits: list[int]) -> list[int]:
        """LeetCode #66: Plus One.

        Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
        The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.
        You may assume the integer does not contain any leading zero, except the number 0 itself.
        """
        # Add 1 to the last digit. Then walk the array backwards, if the current digit is > 9, add 1 to the next digit.
        new = digits
        new[-1] += 1
        for i in range(len(digits)-1,-1,-1): # Walk backwards
            if len(str(digits[i])) != 1: # If the current digit is > 9
                digits[i] = 0 # Set it to 0
                if i == 0: # If it's the first digit
                    new.insert(0,1) # Add a 1
                else:
                    digits[i-1] += 1 # Add 1 to the next digit
            else:
                return new # We're done
        return new

    def addBinary(self, a: str, b: str) -> str:
        """LeetCode #67: Add Binary.

        Given two binary strings a and b, return their sum as a binary string.
        """
        # Get them the same length & walk through the binary math )
        diff = len(a)-len(b) # Get the difference in length
        ans = "" # Start with an empty string
        c = "0" # Set the carry to 0

        if diff > 0: # If a is longer, pad b
            b = "0"*diff+b
        else: # If b is longer, pad a
            a = "0"*-diff+a

        # Iterate through the strings
        for i in range(len(a)-1, -1, -1):
            a1 = a[i] == "1"
            b1 = b[i] == "1"
            c = c == "1"
            ct = sum([a1, b1, c]) # Count the number of 1s

            if  ct >= 3: # If there are 3 or more 1s  # noqa: PLR2004
                ans = "1" + ans
                c = "1"
            elif ct >= 2: # If there are 2 or more 1s  # noqa: PLR2004
                ans = "0" + ans
                c = "1"
            elif ct >= 1: # If there is 1 or more 1s
                ans = "1" + ans
                c = "0"
            else: # If there are no 1s
                ans = "0" + ans
                c = "0"
        if c == "1": # If there is a carry at the end of processing
            ans = "1" + ans
        return ans

    def fullJustify(self, words: list[str], maxWidth: int) -> list[str]:  # noqa: N803
        """LeetCode #68: Text Justification.

        Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
        You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.
        Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
        For the last line of text, it should be left justified and no extra space is inserted between words.
        Note: A word is defined as a character string consisting of non-space characters only.
        """
        # Go through and add the words to the line until you would go over max_width. When you do, generate the line's text value & append it to the answer.
        # For justification, add spaces to the words other than the last one in order until you get to the max_width.
        def justification(words, max_len):
            words_ct = len(words)
            i = 0
            if words_ct == 1: # If there's only one word, special case to make it left justified
                return words[0] + " " * (max_len - len(words[0]))
            while sum([len(word) for word in words]) < max_len: # While the sum of the lengths of the words is less than the max width
                place = i % (words_ct-1) # Get the index of the word to add a space to
                i+=1 # Increment
                words[place] += " " # Add a space
            return "".join(words) # Join the words to form a string

        line_len = 0
        ans = []
        current_line = []
        for word in words: # Iterate through the words
            word_len = len(word)
            if word_len + line_len > maxWidth: # If the line would be too long
                ans.append(justification(current_line, maxWidth)) # Justify the line & add to answer
                current_line = [] # Reset the current line
                line_len = 0 # Reset the line length
            current_line.append(word) # Add the word to the current running list
            line_len += word_len+1 # Add the length of the word + 1 for the space
        # Last line is left justified, so operates differently
        last_line = " ".join(current_line) # Join the words to form a string
        last_line = last_line + " " * (maxWidth-len(last_line)) # Pad the last line
        ans.append(last_line) # Add the last line to the answer
        return ans

    def mySqrt(self, x: int) -> int: #TODO(#10): Solve
        """LeetCode #69: Sqrt(x).

        Given a non-negative integer x, compute and return the square root of x.
        Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.
        """
        n = 1
        while n*n <= x:
            n+=1
        return n-1

    def climbStairs(self, n: int) -> int:
        """LeetCode #70: Climbing Stairs.

        You are climbing a staircase. It takes n steps to reach the top.
        Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
        """
        # 1 - 1 way
        # 2 - 2 ways
        # 3 - 3 ways (1+1+1, 2+1, 1+2)
        # 4 - 5 ways (1+1+1+1, 1+2+1, 2+2, 1+1+2, 2+1+1)
        # It's just expanding the fibonacci sequence
        return util.nth_fibonacci(n)

    def simplifyPath(self, path: str) -> str:
        """LeetCode #71: Simplify Path.

        Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.
        """
        # Split the string by "/", then walk through it
        new_path = r"/"
        for p in path.split("/"):
            if p == "..": # Go up indicator
                new_path = "/".join(new_path.split("/")[:-2]) + "/" # Go up a level
            elif p in {".", ""}: # Ignore these
                continue
            else: # Standard case
                new_path += p + "/"
        return new_path if new_path[-1] != "/" or len(new_path)==1 else new_path[:-1] # Remove trailing slash, unless it's root

    # LeetCode #72: Edit Distance

    def setZeroes(self, matrix: list[list[int]]):
        """LeetCode #73: Set Matrix Zeroes.

        Given an m x n matrix, if an element is 0, set its entire row and column to 0's, then return the matrix.
        You must do it in place.
        """
        # Get the columns & rows to zero in the first pass. Don't do any updates yet, just get the indexes
        # Iterate through the matrix & zero anything in either
        columns_to_zero = []
        rows_to_zero = []

        for row in range(len(matrix)):
            for column in range(len(matrix[row])):
                if matrix[row][column] == 0: # If the element is 0
                    columns_to_zero.append(column) # Add the column
                    rows_to_zero.append(row) # Add the row

        for column in columns_to_zero: # Iterate through the columns that need to be zeroed
            for row in range(len(matrix)): # Iterate through the rows
                matrix[row][column] = 0 # Set the element to 0
        for row in rows_to_zero: # Iterate through the rows that need to be zeroed
            for column in range(len(matrix[row])): # Iterate through the columns
                matrix[row][column] = 0 # Set the element to 0

    # LeetCode #73: Set Matrix Zeroes
    # LeetCode #74: Search a 2D Matrix
    # LeetCode #75: Sort Colors
    # LeetCode #76: Minimum Window Substring
    # LeetCode #77: Combinations
    # LeetCode #78: Subsets
    # LeetCode #79: Word Search
    # LeetCode #80: Remove Duplicates from Sorted Array II
    # LeetCode #81: Search in Rotated Sorted Array II
    # LeetCode #82: Remove Duplicates from Sorted List

    def deleteDuplicates(self, head: [ListNode] | None) -> [ListNode] | None:
        """LeetCode #83: Remove Duplicates from Sorted List.

        Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
        """
        base = ListNode()
        current_node = base
        last_digit = -101 # Somehting we wouldn't see

        if head is None: #Before we get a bunch of errors, check edge cases
            return head

        while head:
            digit = head.val if head else None # Get the next digit to check
            if digit is None:
                return base.next
            if last_digit != digit: # New digit
                current_node.next = ListNode(digit) # Set it
                current_node = current_node.next # Advance it
            head = head.next if head else None #Get the next main list item to check
            last_digit = digit # Update the trailing value

        return base.next


class SolutionButCheeky:
    """Same as Solution, but separated for the cheeky answers."""

    def isMatch(self, s: str, p: str) -> bool:  # TODO(#2): Try without fnmatch
        """LeetCode #44: Wildcard Matching.

        Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.
        '?' Matches any single character.
        '*' Matches any sequence of characters (including the empty sequence).
        The matching should cover the entire input string (not partial).
        """
        return fnmatch.fnmatch(s, p)  # Yeah, i'm feeling cheeky

    def permute(self, nums: list[int]) -> list[list[int]]:  # TODO(#3): Try without itertools
        """LeetCode #46: Permutations.

        Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
        """
        return list(itertools.permutations(nums))  # Easy peasy when you don't care about intention

    def permuteUnique(self, nums: list[int]) -> list[list[int]]:  # TODO(#4): Try without itertools
        """LeetCode #47: Permutations II.

        Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.
        """
        return list(set(itertools.permutations(nums)))  # Easy peasy when you don't care about intention

    def myPow(self, x: float, n: int) -> float:
        """LeetCode #50: Pow(x, n).

        Implement pow(x, n), which calculates x raised to the power n.
        """
        # Did not quite get why this was so hard, it's just x**n?
        if x == 1 or n == 0:
            return 1
        return x**n  # Easy peasy


    def search_alt(self, nums: list[int], target: int) -> int:
        """LeetCode #33: Search in Rotated Sorted Array.

        Original solution
        """
        return (
            nums.index(target) if target in nums else -1
        )  # Okay, so this is maybe overly simple for the intended application, and doesn't meet the complexity requirements (but does pass the submission tests)

class SolutionButAlreadyUsedTheName:
    """Same as Solution, but separated for the already used names."""

    def solveNQueens(self, n: int) -> list[list[str]]:
        solution = []
        row_used = set()
        col_used = set()
        diag1_used = set()
        diag2_used = set()
        board = [
            ["." for i in range(n)] for _ in range(n)
        ]  # Empty board. We're not actually tracking the queens here, it's just for display

        def _brute_force(row):
            if row == n:
                solution.append(["".join(row_list) for row_list in board])  # Add the board to the solution
                return
            for col in range(n):  # Iterate through each column to place the queen
                if (
                    row in row_used or col in col_used or row - col in diag1_used or row + col in diag2_used
                ):  # Check if any row/col/diag already has a queen
                    continue  # skip if so
                board[row][col] = "Q"  # Try placing queen
                # Update the row/col/diags
                row_used.add(row)
                col_used.add(col)
                diag1_used.add(row - col)
                diag2_used.add(row + col)

                # Recursion
                _brute_force(row + 1)

                # Backtrack
                board[row][col] = "."
                row_used.remove(row)
                col_used.remove(col)
                diag1_used.remove(row - col)
                diag2_used.remove(row + col)

        _brute_force(0)  # Start at row 0
        return solution


if __name__ == "__main__":
    sol = Solution()
