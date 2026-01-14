def palindrome(item):
    return str(item) == str(item)[::-1]

def reverse_int(x):
    if x > 0:
        return int(str(x)[::-1])
    return -1*int(str(abs(x))[::-1])

# Definition for singly-linked list.
class ListNode:
     def __init__(self, val=0, next=None):  # noqa: A002
         self.val = val
         self.next = next
