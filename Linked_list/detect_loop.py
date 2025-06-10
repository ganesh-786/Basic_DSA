def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Example 1: Linked list with no cycle: 1 -> 2 -> 3 -> None
head1 = ListNode(1)
head1.next = ListNode(2)
head1.next.next = ListNode(3)

# Example 2: Linked list with a cycle: 1 -> 2 -> 3 -> 2 (cycle)
head2 = ListNode(1)
head2.next = ListNode(2)
head2.next.next = ListNode(3)
head2.next.next.next = head2.next  # Creates a cycle

print("List 1 has cycle:", hasCycle(head1))  # Should print False
print("List 2 has cycle:", hasCycle(head2))  # Should print True