def chunk_iterator(idx, total, chunks, max_length=None):
    '''https://stackoverflow.com/a/37414115'''
    '''If you divide n elements into roughly k chunks you can make n % k chunks 1 element bigger than the other chunks to distribute the extra elements'''
    '''[(n // k) + (1 if i < (n % k) else 0) for i in range(k)]'''

    n = total
    k = chunks

    i0 = idx * (n // k) + min(idx, n % k)
    max_i = (idx + 1) * (n // k) + min(idx + 1, n % k)

    i = i0
    while i < max_i:
        if max_length is not None and i - i0 >= max_length:
            break
        else:
            yield i
            i += 1
