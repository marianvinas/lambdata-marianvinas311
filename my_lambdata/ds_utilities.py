def enlarge(n):
    ''' this function will multiply the input by 100'''
    return n * 100

if __name__ == '__main__':
    y = int(input("Choose a number: "))
    print(y, enlarge(y))