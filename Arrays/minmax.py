arr=[5,6,3,2,1]

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

result=findMinMax(arr)
print(result)