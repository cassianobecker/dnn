def list_from(comma_separated_string):
    return [x.strip() for x in comma_separated_string.split(',')]


def range_to_comma_separated_string(k):
    return ', '.join(str(x) for x in range(k))


def range_to_separated_string(k, token):
    str_token = token + ''
    return str_token.join(str(x) for x in range(k))
