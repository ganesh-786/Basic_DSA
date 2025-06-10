s = 'less'

def reverseString(s):
    # Convert string to list to allow in-place modification
    s_list = list(s)
    left, right = 0, len(s_list) - 1
    while left < right:
        s_list[left], s_list[right] = s_list[right], s_list[left]
        left += 1
        right -= 1
    # Join the list back to a string and return
    return ''.join(s_list)

reversed_s = reverseString(s)
print(reversed_s)