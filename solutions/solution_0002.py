# Definition for singly-linked list.
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
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