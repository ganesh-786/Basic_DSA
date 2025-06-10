# Top 50+ Simple DSA Problems for Interviews

## Arrays

### 1. Search an Element
- **Problem**: Given an array and a target value, return the index of the target if found, else return -1.
- **Solution** (Python):
  ```python
  def searchElement(arr, target):
      for i in range(len(arr)):
          if arr[i] == target:
              return i
      return -1
  ```
- **Explanation**: This problem uses **linear search**, iterating through the array to find the target. The core concept is **array traversal**, a fundamental technique for many array-based problems. It’s simple but sets the stage for understanding more complex search algorithms like binary search.

### 2. Find Minimum and Maximum
- **Problem**: Find the minimum and maximum elements in an array.
- **Solution** (Python):
  ```python
  def findMinMax(arr):
      if not arr:
          return None, None
      min_val, max_val = arr[0], arr[0]
      for num in arr:
          if num < min_val:
              min_val = num
          if num > max_val:
              max_val = num
      return min_val, max_val
  ```
- **Explanation**: Traverse the array once, updating min and max values. This demonstrates **single-pass traversal** with **constant extra space**, a key concept for optimizing array operations.

### 3. Two Sum
- **Problem**: Given an array of integers and a target sum, return the indices of two numbers that add up to the target.
- **Solution** (Python):
  ```python
  def twoSum(nums, target):
      num_map = {}
      for i, num in enumerate(nums):
          complement = target - num
          if complement in num_map:
              return [num_map[complement], i]
          num_map[num] = i
      return []
  ```
- **Explanation**: Use a hash map to store numbers and their indices, checking for the complement in O(1) time. The **hash map for lookups** is a high-impact technique that solves many array problems efficiently.

### 4. Reverse an Array
- **Problem**: Reverse the elements of an array in-place.
- **Solution** (Python):
  ```python
  def reverseArray(arr):
      left, right = 0, len(arr) - 1
      while left < right:
          arr[left], arr[right] = arr[right], arr[left]
          left += 1
          right -= 1
  ```
- **Explanation**: Use two pointers to swap elements from the ends toward the middle. The **two-pointer technique** is a versatile method for in-place array manipulations.

### 5. Move Zeros
- **Problem**: Move all zeros to the end of an array while maintaining the relative order of non-zero elements.
- **Solution** (Python):
  ```python
  def moveZeros(nums):
      left = 0
      for right in range(len(nums)):
          if nums[right] != 0:
              nums[left], nums[right] = nums[right], nums[left]
              left += 1
  ```
- **Explanation**: Use two pointers to place non-zero elements at the front, then fill the rest with zeros. This reinforces the **two-pointer technique** for partitioning arrays.

### 6. Remove Duplicates from Sorted Array
- **Problem**: Remove duplicates from a sorted array in-place and return the new length.
- **Solution** (Python):
  ```python
  def removeDuplicates(nums):
      if not nums:
          return 0
      write_index = 1
      for i in range(1, len(nums)):
          if nums[i] != nums[i - 1]:
              nums[write_index] = nums[i]
              write_index += 1
      return write_index
  ```
- **Explanation**: Use a pointer to track the position for unique elements. This leverages **in-place modification** and the sorted property to achieve O(1) space complexity.

### 7. Rotate Array
- **Problem**: Rotate an array to the right by `k` steps.
- **Solution** (Python):
  ```python
  def rotate(nums, k):
      k = k % len(nums)
      nums[:] = nums[-k:] + nums[:-k]
  ```
- **Explanation**: Use array slicing for rotation. Alternatively, reverse the entire array, then reverse the first `k` and remaining elements. **Array reversal** is a key technique for rotation problems.

## Strings

### 8. Check Anagram
- **Problem**: Check if two strings are anagrams of each other.
- **Solution** (Python):
  ```python
  def isAnagram(s1, s2):
      if len(s1) != len(s2):
          return False
      char_count = {}
      for char in s1:
          char_count[char] = char_count.get(char, 0) + 1
      for char in s2:
          if char not in char_count or char_count[char] == 0:
              return False
          char_count[char] -= 1
      return all(count == 0 for count in char_count.values())
  ```
- **Explanation**: Count character frequencies using a hash map. This highlights **hashing for character counting**, a core concept for string comparison problems.

### 9. Longest Common Prefix
- **Problem**: Find the longest common prefix among an array of strings.
- **Solution** (Python):
  ```python
  def longestCommonPrefix(strs):
      if not strs:
          return ""
      prefix = strs[0]
      for i in range(1, len(strs)):
          while strs[i].find(prefix) != 0:
              prefix = prefix[:-1]
              if not prefix:
                  return ""
      return prefix
  ```
- **Explanation**: Compare the first string with others, shortening the prefix until it matches. **String comparison** is essential for prefix-related problems.

### 10. Reverse String
- **Problem**: Reverse a string in-place.
- **Solution** (Python):
  ```python
  def reverseString(s):
      left, right = 0, len(s) - 1
      while left < right:
          s[left], s[right] = s[right], s[left]
          left += 1
          right -= 1
  ```
- **Explanation**: Use two pointers to swap characters. The **two-pointer technique** applies to both strings and arrays for in-place operations.

### 11. Valid Palindrome
- **Problem**: Check if a string is a palindrome, ignoring non-alphanumeric characters and case.
- **Solution** (Python):
  ```python
  def isPalindrome(s):
      s = ''.join(c.lower() for c in s if c.isalnum())
      return s == s[::-1]
  ```
- **Explanation**: Clean the string and compare it with its reverse. **String preprocessing** and **palindrome checking** are common in string problems.

### 12. First Unique Character
- **Problem**: Find the index of the first non-repeating character in a string.
- **Solution** (Python):
  ```python
  def firstUniqChar(s):
      char_count = {}
      for char in s:
          char_count[char] = char_count.get(char, 0) + 1
      for i, char in enumerate(s):
          if char_count[char] == 1:
              return i
      return -1
  ```
- **Explanation**: Use a hash map to count characters, then find the first with count 1. **Hashing for frequency** is a versatile string technique.

## Linked Lists

### 13. Reverse Linked List
- **Problem**: Reverse a singly linked list.
- **Solution** (Python):
  ```python
  class ListNode:
      def __init__(self, val=0, next=None):
          self.val = val
          self.next = next

  def reverseList(head):
      prev = None
      curr = head
      while curr:
          next_temp = curr.next
          curr.next = prev
          prev = curr
          curr = next_temp
      return prev
  ```
- **Explanation**: Reverse pointers iteratively using three pointers. **Pointer manipulation** is a cornerstone of linked list problems.

### 14. Detect Loop
- **Problem**: Detect if a linked list has a cycle.
- **Solution** (Python):
  ```python
  def hasCycle(head):
      slow = fast = head
      while fast and fast.next:
          slow = slow.next
          fast = fast.next.next
          if slow == fast:
              return True
      return False
  ```
- **Explanation**: Use Floyd’s cycle detection with slow and fast pointers. **Slow and fast pointers** are critical for cycle detection.

### 15. Merge Two Sorted Lists
- **Problem**: Merge two sorted linked lists into one sorted list.
- **Solution** (Python):
  ```python
  def mergeTwoLists(l1, l2):
      dummy = ListNode(0)
      curr = dummy
      while l1 and l2:
          if l1.val < l2.val:
              curr.next = l1
              l1 = l1.next
          else:
              curr.next = l2
              l2 = l2.next
          curr = curr.next
      curr.next = l1 or l2
      return dummy.next
  ```
- **Explanation**: Use a dummy node to build the merged list by comparing nodes. **Dummy nodes** simplify edge cases in linked lists.

### 16. Remove Nth Node From End
- **Problem**: Remove the nth node from the end of a linked list.
- **Solution** (Python):
  ```python
  def removeNthFromEnd(head, n):
      dummy = ListNode(0)
      dummy.next = head
      fast = slow = dummy
      for _ in range(n + 1):
          fast = fast.next
      while fast:
          fast = fast.next
          slow = slow.next
      slow.next = slow.next.next
      return dummy.next
  ```
- **Explanation**: Use two pointers with a gap of `n` to locate the node to remove. **Two-pointer technique** is key for relative positioning.

### 17. Find Middle of Linked List
- **Problem**: Find the middle node of a linked list.
- **Solution** (Python):
  ```python
  def middleNode(head):
      slow = fast = head
      while fast and fast.next:
          slow = slow.next
          fast = fast.next.next
      return slow
  ```
- **Explanation**: Use slow and fast pointers to find the middle. **Slow and fast pointers** are versatile for finding positions.

## Stacks

### 18. Valid Parentheses
- **Problem**: Check if a string of parentheses is valid.
- **Solution** (Python):
  ```python
  def isValid(s):
      stack = []
      brackets = {')': '(', '}': '{', ']': '['}
      for char in s:
          if char in brackets.values():
              stack.append(char)
          elif char in brackets and (not stack or stack.pop() != brackets[char]):
              return False
      return len(stack) == 0
  ```
- **Explanation**: Use a stack to match opening and closing brackets. **Stack (LIFO)** is ideal for pairing problems.

### 19. Min Stack
- **Problem**: Design a stack that supports getMin() in O(1) time.
- **Solution** (Python):
  ```python
  class MinStack:
      def __init__(self):
          self.stack = []
          self.min_stack = []
      def push(self, val):
          self.stack.append(val)
          if not self.min_stack or val <= self.min_stack[-1]:
              self.min_stack.append(val)
      def pop(self):
          if self.stack.pop() == self.min_stack[-1]:
              self.min_stack.pop()
      def top(self):
          return self.stack[-1]
      def getMin(self):
          return self.min_stack[-1]
  ```
- **Explanation**: Use an auxiliary stack to track minimums. **Auxiliary data structures** enhance stack functionality.

### 20. Next Greater Element
- **Problem**: Find the next greater element for each element in an array.
- **Solution** (Python):
  ```python
  def nextGreaterElement(nums):
      stack = []
      result = [-1] * len(nums)
      for i in range(len(nums)):
          while stack and nums[i] > nums[stack[-1]]:
              result[stack.pop()] = nums[i]
          stack.append(i)
      return result
  ```
- **Explanation**: Use a monotonic stack to find the next greater element. **Monotonic stacks** are key for next-greater problems.

## Queues

### 21. Implement Queue using Array
- **Problem**: Implement a queue using an array.
- **Solution** (Python):
  ```python
  class MyQueue:
      def __init__(self):
          self.queue = []
      def enqueue(self, x):
          self.queue.append(x)
      def dequeue(self):
          if not self.queue:
              return None
          return self.queue.pop(0)
      def isEmpty(self):
          return len(self.queue) == 0
  ```
- **Explanation**: Use an array with append for enqueue and pop(0) for dequeue. **Queue (FIFO)** is essential for sequential processing.

### 22. Reverse First K Elements
- **Problem**: Reverse the first `k` elements of a queue.
- **Solution** (Python):
  ```python
  from collections import deque
  def reverseK(queue, k):
      stack = []
      for _ in range(k):
          stack.append(queue.popleft())
      while stack:
          queue.append(stack.pop())
      for _ in range(len(queue) - k):
          queue.append(queue.popleft())
      return queue
  ```
- **Explanation**: Use a stack to reverse the first `k` elements, then rotate the rest. **Stack-queue combination** is a common pattern.

## Trees

### 23. Inorder Traversal
- **Problem**: Perform an inorder traversal of a binary tree.
- **Solution** (Python):
  ```python
  class TreeNode:
      def __init__(self, val=0, left=None, right=None):
          self.val = val
          self.left = left
          self.right = right

  def inorderTraversal(root):
      result = []
      def inorder(node):
          if node:
              inorder(node.left)
              result.append(node.val)
              inorder(node.right)
      inorder(root)
      return result
  ```
- **Explanation**: Recursively visit left, node, then right. **Tree traversals** (inorder, preorder, postorder) are foundational for tree problems.

### 24. Check for BST
- **Problem**: Check if a binary tree is a valid binary search tree (BST).
- **Solution** (Python):
  ```python
  def isValidBST(root, min_val=float('-inf'), max_val=float('inf')):
      if not root:
          return True
      if root.val <= min_val or root.val >= max_val:
          return False
      return isValidBST(root.left, min_val, root.val) and isValidBST(root.right, root.val, max_val)
  ```
- **Explanation**: Recursively check each node against a valid range. **Range validation** ensures BST properties.

### 25. Maximum Depth of Binary Tree
- **Problem**: Find the maximum depth of a binary tree.
- **Solution** (Python):
  ```python
  def maxDepth(root):
      if not root:
          return 0
      return 1 + max(maxDepth(root.left), maxDepth(root.right))
  ```
- **Explanation**: Recursively compute the depth of each subtree. **Recursive tree traversal** is a core concept.

### 26. Symmetric Tree
- **Problem**: Check if a binary tree is symmetric.
- **Solution** (Python):
  ```python
  def isSymmetric(root):
      def isMirror(left, right):
          if not left and not right:
              return True
          if not left or not right:
              return False
          return (left.val == right.val and 
                  isMirror(left.left, right.right) and 
                  isMirror(left.right, right.left))
      return isMirror(root, root)
  ```
- **Explanation**: Recursively compare left and right subtrees symmetrically. **Mirror recursion** is key for symmetry problems.

## Graphs

### 27. DFS of Graph
- **Problem**: Perform a depth-first search (DFS) on a graph.
- **Solution** (Python):
  ```python
  def dfs(graph, start, visited=None):
      if visited is None:
          visited = set()
      visited.add(start)
      result = [start]
      for neighbor in graph[start]:
          if neighbor not in visited:
              result.extend(dfs(graph, neighbor, visited))
      return result
  ```
- **Explanation**: Use recursion to explore as far as possible along each branch. **DFS** is a fundamental graph traversal technique.

### 28. BFS of Graph
- **Problem**: Perform a breadth-first search (BFS) on a graph.
- **Solution** (Python):
  ```python
  from collections import deque
  def bfs(graph, start):
      visited = set()
      queue = deque([start])
      visited.add(start)
      result = []
      while queue:
          node = queue.popleft()
          result.append(node)
          for neighbor in graph[node]:
              if neighbor not in visited:
                  visited.add(neighbor)
                  queue.append(neighbor)
      return result
  ```
- **Explanation**: Use a queue to explore nodes level by level. **BFS** is essential for shortest path and level-order problems.

### 29. Number of Islands
- **Problem**: Count the number of islands in a 2D grid.
- **Solution** (Python):
  ```python
  def numIslands(grid):
      def dfs(i, j):
          if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1':
              return
          grid[i][j] = '0'
          dfs(i+1, j)
          dfs(i-1, j)
          dfs(i, j+1)
          dfs(i, j-1)
      count = 0
      for i in range(len(grid)):
          for j in range(len(grid[0])):
              if grid[i][j] == '1':
                  dfs(i, j)
                  count += 1
      return count
  ```
- **Explanation**: Use DFS to mark connected land as visited. **DFS for connected components** is a key graph technique.

## Sorting

### 30. Merge Two Sorted Arrays
- **Problem**: Merge two sorted arrays into one sorted array.
- **Solution** (Python):
  ```python
  def mergeSortedArrays(arr1, arr2):
      result = []
      i = j = 0
      while i < len(arr1) and j < len(arr2):
          if arr1[i] <= arr2[j]:
              result.append(arr1[i])
              i += 1
          else:
              result.append(arr2[j])
              j += 1
      result.extend(arr1[i:])
      result.extend(arr2[j:])
      return result
  ```
- **Explanation**: Merge arrays by comparing elements and building a new array. **Merge sort logic** is a core sorting concept.

### 31. Kth Smallest Element
- **Problem**: Find the kth smallest element in an unsorted array.
- **Solution** (Python):
  ```python
  import heapq
  def kthSmallest(arr, k):
      return heapq.nsmallest(k, arr)[-1]
  ```
- **Explanation**: Use a heap to find the kth smallest element. **Heaps** are efficient for top-k problems.

## Greedy

### 32. Minimum Platforms
- **Problem**: Find the minimum number of platforms needed for trains given arrival and departure times.
- **Solution** (Python):
  ```python
  def minPlatforms(arrival, departure):
      arrival.sort()
      departure.sort()
      platforms = current = 0
      i = j = 0
      while i < len(arrival):
          if arrival[i] <= departure[j]:
              current += 1
              platforms = max(platforms, current)
              i += 1
          else:
              current -= 1
              j += 1
      return platforms
  ```
- **Explanation**: Sort arrival and departure times, then use two pointers to track platform usage. **Greedy sorting** optimizes resource allocation.

### 33. Fractional Knapsack
- **Problem**: Maximize value in a knapsack with fractional items.
- **Solution** (Python):
  ```python
  def fractionalKnapsack(values, weights, capacity):
      items = sorted([(v/w, w) for v, w in zip(values, weights)], reverse=True)
      total_value = 0
      for value_per_weight, weight in items:
          if capacity >= weight:
              total_value += value_per_weight * weight
              capacity -= weight
          else:
              total_value += value_per_weight * capacity
              break
      return total_value
  ```
- **Explanation**: Sort items by value-to-weight ratio and take as much as possible. **Greedy choice property** is key for optimization problems.

## Recursion

### 34. Print 1 to N
- **Problem**: Print numbers from 1 to n without loops.
- **Solution** (Python):
  ```python
  def print1ToN(n):
      if n == 0:
          return
      print1ToN(n - 1)
      print(n, end=' ')
  ```
- **Explanation**: Use recursion to print numbers. **Recursive base case** and **call stack** are fundamental recursion concepts.

### 35. Factorial
- **Problem**: Compute the factorial of a number.
- **Solution** (Python):
  ```python
  def factorial(n):
      if n == 0 or n == 1:
          return 1
      return n * factorial(n - 1)
  ```
- **Explanation**: Recursively multiply n by factorial of n-1. **Recursive reduction** is a core recursion technique.

## Dynamic Programming

### 36. Climbing Stairs
- **Problem**: Find the number of ways to climb `n` stairs with 1 or 2 steps.
- **Solution** (Python):
  ```python
  def climbStairs(n):
      if n <= 2:
          return n
      dp = [0] * (n + 1)
      dp[1], dp[2] = 1, 2
      for i in range(3, n + 1):
          dp[i] = dp[i - 1] + dp[i - 2]
      return dp[n]
  ```
- **Explanation**: Use DP to sum ways from previous steps. **State transition** is a cornerstone of DP.

### 37. 0-1 Knapsack
- **Problem**: Maximize value in a knapsack with items that can’t be split.
- **Solution** (Python):
  ```python
  def knapsack(values, weights, capacity):
      n = len(values)
      dp = [[0] * (capacity + 1) for _ in range(n + 1)]
      for i in range(1, n + 1):
          for w in range(capacity + 1):
              if weights[i - 1] <= w:
                  dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
              else:
                  dp[i][w] = dp[i - 1][w]
      return dp[n][capacity]
  ```
- **Explanation**: Use a 2D DP table to track maximum value for each capacity. **Optimal substructure** drives DP solutions.

### 38. Longest Common Subsequence
- **Problem**: Find the length of the longest common subsequence between two strings.
- **Solution** (Python):
  ```python
  def longestCommonSubsequence(text1, text2):
      m, n = len(text1), len(text2)
      dp = [[0] * (n + 1) for _ in range(m + 1)]
      for i in range(1, m + 1):
          for j in range(1, n + 1):
              if text1[i - 1] == text2[j - 1]:
                  dp[i][j] = dp[i - 1][j - 1] + 1
              else:
                  dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
      return dp[m][n]
  ```
- **Explanation**: Use a 2D DP table to build the subsequence. **Subproblem dependency** is key in DP.

### 39. Fibonacci Number
- **Problem**: Compute the nth Fibonacci number.
- **Solution** (Python):
  ```python
  def fib(n):
      if n <= 1:
          return n
      dp = [0] * (n + 1)
      dp[1] = 1
      for i in range(2, n + 1):
          dp[i] = dp[i - 1] + dp[i - 2]
      return dp[n]
  ```
- **Explanation**: Use DP to avoid redundant calculations. **Memoization** optimizes recursive problems.

## Additional Problems (to reach 50+)

### Arrays
40. **Contains Duplicate**: Check if an array has duplicates using a hash set.
41. **Maximum Subarray**: Find the contiguous subarray with the largest sum using Kadane’s algorithm.
42. **Merge Intervals**: Merge overlapping intervals after sorting by start time.
43. **Product of Array Except Self**: Compute product of all elements except the current one using two passes.
44. **Find Peak Element**: Find a peak element using binary search.

### Strings
45. **Longest Substring Without Repeating Characters**: Use a sliding window with a hash set.
46. **String to Integer (atoi)**: Parse a string to an integer with boundary checks.
47. **Count and Say**: Generate the nth term of the count-and-say sequence recursively.
48. **Group Anagrams**: Group strings by their sorted characters using a hash map.

### Linked Lists
49. **Add Two Numbers**: Add two numbers represented as linked lists with carry tracking.
50. **Intersection of Two Linked Lists**: Find the intersection node using pointer alignment.
51. **Palindrome Linked List**: Check if a linked list is a palindrome using slow and fast pointers.

### Stacks
52. **Implement Queue using Stacks**: Use two stacks to simulate a queue.
53. **Evaluate Reverse Polish Notation**: Use a stack to process postfix expressions.

### Queues
54. **Implement Stack using Queues**: Use a queue to simulate a stack by reordering elements.

### Trees
55. **Preorder Traversal**: Traverse a binary tree in preorder (root, left, right).
56. **Postorder Traversal**: Traverse a binary tree in postorder (left, right, root).
57. **Lowest Common Ancestor**: Find the LCA of two nodes using recursive traversal.

### Graphs
58. **Clone Graph**: Clone a graph using DFS with a hash map.
59. **Course Schedule**: Detect cycles in a graph using topological