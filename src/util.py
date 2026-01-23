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


def binary_search(arr, target, which="any", not_found=-1):
    """Perform a binary search on a sorted list to find the index of a target value.

    Args:
        arr (list): A sorted list of elements to search.
        target: The value to search for in the list.
        which (str, optional): Specifies which occurrence to find:
            - "any": Returns the index of any occurrence of the target.
            - "first": Returns the index of the first occurrence of the target.
            - "last": Returns the index of the last occurrence of the target.
            Defaults to "any".
        not_found (int, optional): The value to return if the target is not found.
            Defaults to -1.

    Returns:
        int: The index of the found target value according to the 'which' parameter.
             If the target is not found, returns not_found.

    """
    left_point = 0
    right_point = len(arr) - 1
    ix = not_found

    while left_point <= right_point:
        mid = (left_point + right_point)//2

        if arr[mid] == target: # Found the target
            ix = mid # Update index value with latest binary search result
            if which.lower() == "first": # If you're looking for first occurrence, move the right side over
                right_point = mid-1
            elif which.lower() == "last": # If you're looking for last occurrence, move the left side over
                left_point = mid+1
            else: # Any element will do
                return ix
        elif target < arr[mid]: # Need to move earlier in list
            right_point = mid-1
        else: # Need to move later in list
            left_point = mid + 1
    return ix # Return the last found index
