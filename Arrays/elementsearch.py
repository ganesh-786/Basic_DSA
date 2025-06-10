arr=[1,2,3,4,5]
target=1
def searchElement(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

result = searchElement(arr, target)
print(result)