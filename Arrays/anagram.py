
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


print(isAnagram(s1="mess",s2="ssem"))