import time


def problem3(s):
    max_val = 0
    if s == "":
        return 0
    if len(s) == len(set(s)):
        return len(s)
    for i in range(len(s)):
        for j in range(i + 1 + max_val, len(s)+1):
            if len(set(s[i:j])) == len(s[i:j]):
                max_val = max(max_val, len(s[i:j]))
    return max_val


if __name__ == "__main__":
    """[summary]"""
    start_time = time.time()
    print(problem3("mpesjbcxgdfucjbrazpzpzdrlnepyiikzoeirghxkmsoytgyuxxjycdmqhbqrjasyhapnkpzyjowewuztt"), time.time() - start_time)
