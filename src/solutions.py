from __future__ import annotations

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

    def search_alt(self, nums: list[int], target: int) -> int:
        """LeetCode #33: Search in Rotated Sorted Array.

        Original solution
        """
        return (
            nums.index(target) if target in nums else -1
        )  # Okay, so this is maybe overly simple for the intended application, and doesn't meet the complexity requirements (but does pass the submission tests)

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


if __name__ == "__main__":
    sol = Solution()
